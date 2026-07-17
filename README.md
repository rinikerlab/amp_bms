# AMP-BMS

## About
AMP-BMS is a multiscale neural network potential trained on QM/MM data with electrostatic embedding for the simulations of biomolecules in the condensed phase. This repo was used for our [recent publication]() together with a [dataset for multiscale neural networks](https://www.research-collection.ethz.ch/entities/researchdata/15515ac0-d9a6-4966-b658-5e391907ef43).
This model is based on the [AMP architecture](https://openreview.net/forum?id=socffUzSIlx) and was used in previous work [1](https://pubs.rsc.org/en/content/articlehtml/2023/sc/d3sc04317g), [2](https://pubs.acs.org/doi/full/10.1021/jacs.4c17015).

## Usage

Example usage is shown in [here](source/examples/examples_md/example-simulation.ipynb).

## Installation

````
# Install environment.
conda env create -f amp.yml
conda activate amp_bms

# Install repo after cloning from github
pip install -e .
````

## References

If you use this code, please cite our papers:

```bibtex
@article{AMP3,
  title = {Multiscale Neural Network Potential with Anisotropic Message Passing for the Fast and Accurate Simulation of Protein Dynamics and Enzymatic Reactions},
  volume = {148},
  ISSN = {1520-5126},
  url = {http://dx.doi.org/10.1021/jacs.6c00217},
  DOI = {10.1021/jacs.6c00217},
  number = {27},
  journal = {Journal of the American Chemical Society},
  publisher = {American Chemical Society (ACS)},
  author = {Th\"{u}rlemann,  Moritz and Pultar,  Felix and Gordiy,  Igor and Ruijsenaars,  Enrico and Riniker,  Sereina},
  year = {2026},
  pages = {28133--28156}

@article{AMP2,
  title = {Neural Network Potential with Multiresolution Approach Enables Accurate Prediction of Reaction Free Energies in Solution},
  volume = {147},
  ISSN = {1520-5126},
  url = {http://dx.doi.org/10.1021/jacs.4c17015},
  DOI = {10.1021/jacs.4c17015},
  number = {8},
  journal = {Journal of the American Chemical Society},
  publisher = {American Chemical Society (ACS)},
  author = {Pultar,  Felix and Th\"{u}rlemann,  Moritz and Gordiy,  Igor and Doloszeski,  Eva and Riniker,  Sereina},
  year = {2025},
  pages = {6835--6856}
}
}
```

## Contributors
Moritz Thürlemann  
Felix Pultar  
Igor Gordiy  

## License

The AMP-BMS code is published and distributed under the [MIT License](LICENSE). 




