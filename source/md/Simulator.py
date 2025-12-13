import logging
import os
from sys import stdout

import numpy as np
import openmm as mm
import torch

from openff.toolkit import Molecule, Topology
from openmm import unit as u
from openmm.app import PDBFile
from openmmforcefields.generators import SystemGenerator
from openmmtorch import TorchForce

from amp.AMP import AMP
from md.TorchForce import ForceModule
from utilities.Helpers import load_parameters


class Simulator:
    def __init__(
        self,
        T=300 * u.kelvin,
        P=1 * u.bar,
        cutoff_nb=9 * u.angstrom,  # Default cutoff for TIP4P-FB´
        padding=10 * u.angstrom,
        friction=1 / u.picosecond,
        dT=0.5 * u.femtoseconds,
        add_hydrogens=True,
        integrator=None,
        constraints=None,
        add_solvent=True,
        model_type="MIN",
        ff_protein="amber14-all.xml",
        ff_molecules="openff-2.2.0",
        ff_water="amber14/tip4pfb.xml",
        ff_additional=None,
        water_model="tip4pew",
        output_folder=None,
        add_state_reporter=True,
        add_dcd_reporter=True,
        add_chk_reporter=True,
        report_frequency=1000,
        device_mm="cuda",
        device_ml="cuda",
        neutralize_box=False,
        eps_rf=78.4,
        baro_freq=100,
        use_baro=True,
        vdw_correction=True,
        pH=7.0,
        restrain_solute=False,
        restrain_ligand=False,
        n_nlist=64,
        pairlist_padding=4.0,
        chunk_size=10000000,
        block_size=4000,
        output_suffix="",
        use_exclusions=True,
        **kwargs,
    ):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        self.T = T
        self.P = P
        self.pH = pH
        self.dT = dT
        self.restrain_solute = restrain_solute
        self.restrain_ligand = restrain_ligand
        self.add_hydrogens = add_hydrogens
        self.constraints = constraints
        self.add_solvent = add_solvent
        self.platform = mm.Platform.getPlatformByName(device_mm.upper())
        self.device_mm = device_mm
        self.device_ml = device_ml
        self.friction = friction
        self.model_type = model_type
        self.ff_protein = ff_protein
        self.ff_molecules = ff_molecules
        self.ff_water = ff_water
        self.ff_additional = ff_additional
        self.water_model = water_model
        self.neutralize_box = neutralize_box
        self.eps_rf = eps_rf
        self.use_baro = use_baro
        self.vdw_correction = vdw_correction
        self.baro_freq = baro_freq
        self.output_folder = output_folder
        self.add_state_reporter = add_state_reporter
        self.add_dcd_reporter = add_dcd_reporter
        self.add_chk_reporter = add_chk_reporter
        self.report_frequency = report_frequency
        self.integrator = integrator
        self.n_nlist = n_nlist
        self.pairlist_padding = pairlist_padding
        self.chunk_size = chunk_size
        self.block_size = block_size
        self.output_suffix = output_suffix
        self.use_exclusions = use_exclusions
        self.cutoff_nb = cutoff_nb
        self.padding = padding
        self.logger.info(
            f"Using a cutoff of {self.cutoff_nb} A for the nonbonded interactions"
        )
        self.logger.info(
            f"Generating pairlist every {self.n_nlist} steps with {self.pairlist_padding} A padding."
        )

    def __call__(
        self,
        pdb_input=None,
        molecules=None,
        minimize=True,
        max_iterations=1000,
        fix=True,
        restart=False,
    ):
        assert (
            pdb_input is not None or molecules is not None
        ), "specify either a pdb with a protein, a list of openff toolkit molecules, or both"
        if pdb_input:
            assert isinstance(
                pdb_input, str
            ), "pdb_input must refer to a valid pdb file"
        if molecules:
            assert isinstance(molecules, list) and all(
                isinstance(mol, Molecule) for mol in molecules
            ), "molecules must be a list of openff toolkit molecules"

        if restart:
            self.logger.info("Restart requested...")
        else:
            self.logger.info("No restart requested...")

        if pdb_input:
            self.logger.info(f"Loading {pdb_input}...")
            system, modeller = self._initialize_system_from_pdb(
                protein_input=pdb_input, molecules=molecules, fix=fix, restart=restart
            )
        else:
            system, modeller = self._initialize_system_from_molecules(
                molecules=molecules, restart=restart
            )

        self._get_topo_indices(modeller.topology)
        self._extract_charges(system)
        if self.restrain_solute:
            self._remove_masses(system)
        if self.restrain_ligand:
            self._remove_masses_ligand(system, modeller)
        system = self._modify_forces(system, modeller)
        self.modeller, self.system = modeller, system
        torch_force = self._build_torchforce()
        self.system.addForce(torch_force)
        if self.use_baro:
            self.system.addForce(mm.MonteCarloBarostat(self.P, self.T, self.baro_freq))
        if self.integrator is None:
            integrator = mm.LangevinMiddleIntegrator(self.T, self.friction, self.dT)
        simulation = mm.app.Simulation(
            self.modeller.topology, self.system, integrator, self.platform
        )
        simulation.context.setPositions(self.modeller.positions)
        if minimize and not restart:
            self.logger.info(f"Start Minimization ...")
            self.logger.info(
                f"{simulation.context.getState(getEnergy=True).getPotentialEnergy()}"
            )
            simulation.minimizeEnergy(maxIterations=max_iterations)
            self.logger.info(
                f"Minimization Converged: {simulation.context.getState(getEnergy=True).getPotentialEnergy()}"
            )
        simulation = self._append_reporters(simulation, restart=restart)
        self.simulation = simulation
        return simulation

    def _process_pdb_file(self, input_file, fix, restart):
        if fix and not restart:
            from pdbfixer import PDBFixer

            pdbfixer = PDBFixer(input_file)
            pdbfixer.removeHeterogens(False)
            pdbfixer.missingResidues = {}
            pdbfixer.findMissingAtoms()
            pdbfixer.findMissingResidues()
            pdbfixer.findNonstandardResidues()
            pdbfixer.replaceNonstandardResidues()
            pdbfixer.addMissingAtoms()
            return pdbfixer
        else:
            return mm.app.PDBFile(input_file)

    def _initialize_system_from_pdb(self, protein_input, molecules, fix, restart):
        self.logger.info("Initializing system with protein...")
        protein_pdb = self._process_pdb_file(protein_input, fix=fix, restart=restart)
        forcefield = (
            mm.app.ForceField(self.ff_protein, self.ff_water)
            if self.ff_additional == None
            else mm.app.ForceField(self.ff_protein, self.ff_water, self.ff_additional)
        )
        modeller = mm.app.Modeller(protein_pdb.topology, protein_pdb.positions)

        # add small molecule if present
        if molecules:
            forcefield_kwargs = {"constraints": self.constraints, "rigidWater": True}
            periodic_forcefield_kwargs = {"nonbondedMethod": mm.app.PME}
            forcefields = [self.ff_protein, self.ff_water]
            if self.ff_additional:
                forcefields.append(self.ff_additional)
            system_generator = SystemGenerator(
                forcefields=forcefields,
                small_molecule_forcefield=self.ff_molecules,
                molecules=molecules,
                forcefield_kwargs=forcefield_kwargs,
                nonperiodic_forcefield_kwargs=periodic_forcefield_kwargs,
            )
            if not restart:
                for molecule in molecules:
                    modeller.add(
                        molecule.to_topology().to_openmm(),
                        molecule.to_topology().get_positions().to_openmm(),
                    )
            forcefield = system_generator.forcefield

        if not restart:
            if self.add_hydrogens:
                modeller.addHydrogens(forcefield, pH=self.pH)
            if self.add_solvent:
                modeller.addSolvent(
                    forcefield,
                    model=self.water_model,
                    padding=self.padding,
                    neutralize=self.neutralize_box,
                )
            modeller.addExtraParticles(forcefield)

        system = forcefield.createSystem(
            modeller.topology,
            constraints=self.constraints,
            rigidWater=True,
            nonbondedMethod=mm.app.PME,
        )
        box = np.array(
            [
                np.array(x.value_in_unit(u.angstrom))
                for x in system.getDefaultPeriodicBoxVectors()
            ]
        )
        ortho = (box[np.tril_indices(3, k=-1)] == 0).all() and (
            box[np.triu_indices(3, k=1)] == 0
        ).all()
        assert ortho, "Currently PBC are only implemented for orthorhombic boxes."
        return system, modeller

    def _initialize_system_from_molecules(self, molecules, restart):
        self.logger.info("Initializing system with small molecules...")
        topology = Topology.from_molecules(molecules)
        system_generator = SystemGenerator(
            forcefields=[self.ff_water],
            small_molecule_forcefield=self.ff_molecules,
            molecules=molecules,
        )
        modeller = mm.app.Modeller(
            topology.to_openmm(), topology.get_positions().to_openmm()
        )
        forcefield = system_generator.forcefield

        if not restart:
            if self.add_hydrogens:
                modeller.addHydrogens(forcefield, pH=self.pH)
            if self.add_solvent:
                modeller.addSolvent(
                    forcefield,
                    model=self.water_model,
                    padding=self.padding,
                    neutralize=self.neutralize_box,
                )
            modeller.addExtraParticles(forcefield)

        system = forcefield.createSystem(
            modeller.topology,
            constraints=self.constraints,
            rigidWater=True,
            nonbondedMethod=mm.app.PME,
        )

        box = np.array(
            [
                np.array(x.value_in_unit(u.angstrom))
                for x in system.getDefaultPeriodicBoxVectors()
            ]
        )
        ortho = (box[np.tril_indices(3, k=-1)] == 0).all() and (
            box[np.triu_indices(3, k=1)] == 0
        ).all()
        assert ortho, "Currently PBC are only implemented for orthorhombic boxes."
        return system, modeller

    def _build_torchforce(self):
        module_dir = os.path.dirname(os.path.abspath(__file__))
        model_weights = os.path.join(
            module_dir, "..", "model_weights", f"{self.model_type}_state_dict"
        )
        self.logger.info(f"Loaded model weights for {self.model_type}")
        param_file = os.path.join(
            module_dir, "..", "parameters", f"PARAMETERS_{self.model_type}.yaml"
        )
        PARAMETERS = load_parameters(param_file)
        model = AMP(PARAMETERS).to(self.device_ml)
        self.logger.info(f"Model: Graph cutoff: {model.cutoff:.1f} A")
        self.logger.info(f"Model: ESP cutoff: {model.cutoff_esp:.1f} A")
        self.logger.info(f"Model: QM/MM ESP cutoff: {model.cutoff_qmmm_esp:.1f} A")
        self.logger.info(
            f"Model: QM/MM polarization cutoff: {model.cutoff_qmmm_pol:.1f} A"
        )
        model.load_state_dict(
            torch.load(model_weights, map_location=self.device_ml, weights_only=True)
        )
        torch.jit.enable_onednn_fusion(True)
        model_scripted = torch.jit.script(model)
        self.logger.info(f"Initialized Model.")
        force_module = ForceModule(
            model_scripted,
            self,
            device=self.device_ml,
            n_nlist=self.n_nlist,
            pairlist_padding=self.pairlist_padding,
            chunk_size=self.chunk_size,
            block_size=self.block_size,
        )
        module = torch.jit.script(force_module).to(self.device_ml)
        torch_force = TorchForce(module)
        torch_force.setUsesPeriodicBoundaryConditions(True)
        self.logger.info(
            f"Running ML model on {self.device_ml} and OpenMM on {self.device_mm} with single precision."
        )
        return torch_force

    def _append_reporters(self, simulation, restart):
        if self.output_folder is None:
            output_folder = "trajectory_output"
        else:
            output_folder = self.output_folder
        if self.add_dcd_reporter or self.add_chk_reporter or self.add_state_reporter:
            try:
                os.mkdir(output_folder)
            except:
                pass
        if not restart:
            positions = simulation.context.getState(getPositions=True).getPositions()
            PDBFile.writeFile(
                simulation.topology,
                positions,
                open(f"{output_folder}init_topology{self.output_suffix}.pdb", "w"),
            )
        if self.add_dcd_reporter:
            simulation.reporters.append(
                mm.app.DCDReporter(
                    f"{output_folder}trajectory{self.output_suffix}.dcd",
                    self.report_frequency,
                )
            )
        if self.add_state_reporter:
            simulation.reporters.append(
                mm.app.StateDataReporter(
                    stdout,
                    self.report_frequency,
                    step=True,
                    potentialEnergy=True,
                    temperature=True,
                    volume=True,
                    density=True,
                    speed=True,
                    separator="\t",
                )
            )
            simulation.reporters.append(
                mm.app.StateDataReporter(
                    f"{output_folder}trajectory{self.output_suffix}.csv",
                    self.report_frequency,
                    step=True,
                    potentialEnergy=True,
                    temperature=True,
                    volume=True,
                    density=True,
                    speed=True,
                    separator="\t",
                )
            )
        if self.add_chk_reporter:
            simulation.reporters.append(
                mm.app.CheckpointReporter(
                    f"{output_folder}checkpoint.xml",
                    self.report_frequency,
                    writeState=True,
                )
            )
        return simulation

    def _modify_forces(self, system, modeller):
        for idf, force in enumerate(system.getForces()):
            if isinstance(force, mm.openmm.HarmonicBondForce):
                bond_force = force
            if isinstance(force, mm.openmm.NonbondedForce):
                nb_force = force
        if self.use_exclusions:
            force_lj, force_crf = self.build_custom_nonbonded_mmmm_exclusions(nb_force)
        else:
            force_lj, force_crf = self.build_custom_nonbonded_mmmm_no_exclusions(
                nb_force
            )
        bond_force = self._modify_bond_force(bond_force)
        forces = system.getForces()
        while len(forces) > 2:
            forces = system.getForces()
            for idf, force in enumerate(forces):
                if not isinstance(force, mm.openmm.CMMotionRemover) and not isinstance(
                    force, mm.openmm.HarmonicBondForce
                ):
                    system.removeForce(idf)
                    break
        system.addForce(force_lj)
        system.addForce(force_crf)
        return system

    def _modify_bond_force(self, bond_force):
        for bond_id in range(bond_force.getNumBonds()):
            bond_id_1, bond_id_2, eq_distance, k_constant = (
                bond_force.getBondParameters(bond_id)
            )
            bond_force.setBondParameters(
                bond_id, bond_id_1, bond_id_2, eq_distance, k_constant * 0
            )
        return bond_force

    def _get_topo_indices(self, topology):
        if "TIP4P" in self.water_model.upper():
            charged_mm_atoms = ["H1", "H2", "M"]
        elif "TIP3P" in self.water_model.upper():
            charged_mm_atoms = ["H1", "H2", "O"]
        qm_zone, mm_zone, mm_zone_charges, water_zone, ion_zone, m_zone = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for atom in topology.atoms():
            resname = atom.residue.name.upper()
            if resname == "HOH":
                mm_zone.append(atom.index)
                water_zone.append(atom.index)
                if atom.name in charged_mm_atoms:
                    mm_zone_charges.append(atom.index)
                if atom.name in ["M"]:
                    m_zone.append(atom.index)
            elif resname in ["CL", "NA", "F", "K", "MG", "CA", "ZN"]:
                mm_zone.append(atom.index)
                ion_zone.append(atom.index)
                mm_zone_charges.append(atom.index)
            else:
                qm_zone.append(atom.index)
        self.qm_zone = np.array(qm_zone)
        self.mm_zone = np.array(mm_zone)
        self.mm_zone_charges = np.array(mm_zone_charges)
        self.water_zone = np.array(water_zone)
        self.ion_zone = np.array(ion_zone)
        self.m_zone = np.array(m_zone)
        n_mm = len(self.mm_zone)
        n_qm = len(self.qm_zone)
        self.logger.info(
            f"Initialized systems with {n_qm} atoms in the QM zone and {n_mm} atoms in the MM zone."
        )

    def _extract_charges(self, system):
        for idf, force in enumerate(system.getForces()):
            if isinstance(force, mm.openmm.NonbondedForce) or isinstance(
                force, mm.openmm.CustomNonbondedForce
            ):
                nb_force = force
        self.charges_mm = np.array(
            [
                nb_force.getParticleParameters(index)[0]._value
                for index in self.mm_zone_charges
            ]
        )
        self.charges_qm = np.array(
            [nb_force.getParticleParameters(index)[0]._value for index in self.qm_zone]
        )
        self.mol_charge = np.round(self.charges_qm.sum())
        self.qm_masses = np.array(
            [system.getParticleMass(index) for index in self.qm_zone]
        )

    def _remove_masses(self, system):
        for index in self.qm_zone:
            system.setParticleMass(index, 0)
        return system

    def _remove_masses_ligand(self, system, modeller):
        for atom in modeller.topology.atoms():
            if atom.residue.name == "UNK":
                self.logger.info(f"setting mass to 0: {atom.index}")
                system.setParticleMass(atom.index, 0)
        return system

    # RF Nonbonded from Salome:
    # https://github.com/rinikerlab/reeds/blob/main/reeds/openmm/reeds_openmm.py
    def build_custom_nonbonded_mmmm_exclusions(self, nb_force):
        self.logger.info("Using exclusion list for custom nonbonded force.")
        krf = ((self.eps_rf - 1) / (1 + 2 * self.eps_rf)) * (1 / self.cutoff_nb**3)
        ONE_4PI_EPS0 = 138.935456  # * u.kilojoules_per_mole*u.nanometer/(u.elementary_charge_base_unit*u.elementary_charge_base_unit)
        mrf = 4
        nrf = 6
        arfm = (3 * self.cutoff_nb ** (-(mrf + 1)) / (mrf * (nrf - mrf))) * (
            (2 * self.eps_rf + nrf - 1) / (1 + 2 * self.eps_rf)
        )
        arfn = (3 * self.cutoff_nb ** (-(nrf + 1)) / (nrf * (mrf - nrf))) * (
            (2 * self.eps_rf + mrf - 1) / (1 + 2 * self.eps_rf)
        )
        crf = (
            ((3 * self.eps_rf) / (1 + 2 * self.eps_rf)) * (1 / self.cutoff_nb)
            + arfm * self.cutoff_nb**mrf
            + arfn * self.cutoff_nb**nrf
        )

        crf_exp = "ONE_4PI_EPS0*chargeprod*(1/r + krf*r2 + arfm*r4 + arfn*r6 - crf);"
        crf_exp += "krf = {:f};".format(krf.value_in_unit(u.nanometer**-3))
        crf_exp += "crf = {:f};".format(crf.value_in_unit(u.nanometer**-1))
        crf_exp += "r6 = r2*r4;"
        crf_exp += "r4 = r2*r2;"
        crf_exp += "r2 = r*r;"
        crf_exp += "arfm = {:f};".format(arfm.value_in_unit(u.nanometer**-5))
        crf_exp += "arfn = {:f};".format(arfn.value_in_unit(u.nanometer**-7))
        crf_exp += "chargeprod = charge1*charge2;"
        crf_exp += "ONE_4PI_EPS0 = {:f};".format(ONE_4PI_EPS0)
        lj_exp = "4*epsilon*(sigma_over_r12 - sigma_over_r6);"
        lj_exp += "sigma_over_r12 = sigma_over_r6 * sigma_over_r6;"
        lj_exp += "sigma_over_r6 = sigma_over_r3 * sigma_over_r3;"
        lj_exp += "sigma_over_r3 = sigma_over_r * sigma_over_r * sigma_over_r;"
        lj_exp += "sigma_over_r = sigma/r;"
        lj_exp += "epsilon = sqrt(epsilon1*epsilon2);"
        lj_exp += "sigma = 0.5*(sigma1+sigma2);"
        force_crf = mm.CustomNonbondedForce(crf_exp)
        force_crf.addPerParticleParameter("charge")
        force_crf.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
        force_crf.setCutoffDistance(self.cutoff_nb)
        force_lj = mm.CustomNonbondedForce(lj_exp)
        force_lj.addPerParticleParameter("sigma")
        force_lj.addPerParticleParameter("epsilon")
        force_lj.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
        force_lj.setCutoffDistance(self.cutoff_nb)
        force_lj.setUseLongRangeCorrection(self.vdw_correction)
        for index in range(nb_force.getNumParticles()):
            charge, sigma, epsilon = nb_force.getParticleParameters(index)
            # Set charges within qm zone to zero as electrostatics are
            # handled by the ML potential.
            if index in self.qm_zone:
                charge = 0
            force_crf.addParticle([charge])
            force_lj.addParticle([sigma, epsilon])
        # Retain exceptions for mm-mm interactions
        for index in range(nb_force.getNumExceptions()):
            j, k, chargeprod, sigma, epsilon = nb_force.getExceptionParameters(index)
            if j in self.mm_zone and k in self.mm_zone:
                force_lj.addExclusion(j, k)
                force_crf.addExclusion(j, k)
        # Adds exceptions for qm-qm interactions
        for j in self.qm_zone:
            for k in self.qm_zone[j + 1 :]:
                force_lj.addExclusion(j, k)
                force_crf.addExclusion(j, k)
        return force_lj, force_crf

    def build_custom_nonbonded_mmmm_no_exclusions(self, nb_force):
        self.logger.info("Using no exclusion list for custom nonbonded force.")
        krf = ((self.eps_rf - 1) / (1 + 2 * self.eps_rf)) * (1 / self.cutoff_nb**3)
        ONE_4PI_EPS0 = 138.935456  # * u.kilojoules_per_mole*u.nanometer/(u.elementary_charge_base_unit*u.elementary_charge_base_unit)
        mrf = 4
        nrf = 6
        arfm = (3 * self.cutoff_nb ** (-(mrf + 1)) / (mrf * (nrf - mrf))) * (
            (2 * self.eps_rf + nrf - 1) / (1 + 2 * self.eps_rf)
        )
        arfn = (3 * self.cutoff_nb ** (-(nrf + 1)) / (nrf * (mrf - nrf))) * (
            (2 * self.eps_rf + mrf - 1) / (1 + 2 * self.eps_rf)
        )
        crf = (
            ((3 * self.eps_rf) / (1 + 2 * self.eps_rf)) * (1 / self.cutoff_nb)
            + arfm * self.cutoff_nb**mrf
            + arfn * self.cutoff_nb**nrf
        )

        crf_exp = "ONE_4PI_EPS0*chargeprod*(1/r + krf*r2 + arfm*r4 + arfn*r6 - crf);"
        crf_exp += "krf = {:f};".format(krf.value_in_unit(u.nanometer**-3))
        crf_exp += "crf = {:f};".format(crf.value_in_unit(u.nanometer**-1))
        crf_exp += "r6 = r2*r4;"
        crf_exp += "r4 = r2*r2;"
        crf_exp += "r2 = r*r;"
        crf_exp += "arfm = {:f};".format(arfm.value_in_unit(u.nanometer**-5))
        crf_exp += "arfn = {:f};".format(arfn.value_in_unit(u.nanometer**-7))
        crf_exp += "chargeprod = charge1*charge2;"
        crf_exp += "ONE_4PI_EPS0 = {:f};".format(ONE_4PI_EPS0)

        lj_exp = "4*epsilon*(sigma_over_r12 - sigma_over_r6) * ((isMM1+isMM2) - (isMM1*isMM2));"
        lj_exp += "sigma_over_r12 = sigma_over_r6 * sigma_over_r6;"
        lj_exp += "sigma_over_r6 = sigma_over_r3 * sigma_over_r3;"
        lj_exp += "sigma_over_r3 = sigma_over_r * sigma_over_r * sigma_over_r;"
        lj_exp += "sigma_over_r = sigma/r;"
        lj_exp += "epsilon = sqrt(epsilon1*epsilon2);"
        lj_exp += "sigma = 0.5*(sigma1+sigma2);"

        force_crf = mm.CustomNonbondedForce(crf_exp)
        force_crf.addPerParticleParameter("charge")
        force_crf.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
        force_crf.setCutoffDistance(self.cutoff_nb)

        force_lj = mm.CustomNonbondedForce(lj_exp)
        force_lj.addPerParticleParameter("sigma")
        force_lj.addPerParticleParameter("epsilon")
        force_lj.addPerParticleParameter("isMM")
        force_lj.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
        force_lj.setCutoffDistance(self.cutoff_nb)
        force_lj.setUseLongRangeCorrection(self.vdw_correction)

        mm_zone_set = set(self.mm_zone)
        for index in range(nb_force.getNumParticles()):
            charge, sigma, epsilon = nb_force.getParticleParameters(index)
            if index not in mm_zone_set:
                isMM = 0  # logical OR implemented in LJ (electrosatic embedding)
                charge = 0  # QM/MM CRF should evaluate to 0 (electrosatic embedding)
            else:
                isMM = 1
            force_crf.addParticle([charge])
            force_lj.addParticle([sigma, epsilon, isMM])

        # Retain exceptions for mm-mm interactions
        for index in range(nb_force.getNumExceptions()):
            j, k, _, sigma, epsilon = nb_force.getExceptionParameters(index)
            if j in mm_zone_set and k in mm_zone_set:
                force_lj.addExclusion(j, k)
                force_crf.addExclusion(j, k)

        return force_lj, force_crf
