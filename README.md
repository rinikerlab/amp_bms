# AMP-BMS

## About
AMP-BMS is a multiscale neural network potential trained on QM/MM data with electrostatic embedding for the simulations of biomolecules in the condensed phase. This repo was used for our [recent publication]() together with a [dataset for multiscale neural networks](https://www.research-collection.ethz.ch/entities/researchdata/15515ac0-d9a6-4966-b658-5e391907ef43).
This model is based on the [AMP architecture](https://openreview.net/forum?id=socffUzSIlx) and was used in previous work [1](https://pubs.rsc.org/en/content/articlehtml/2023/sc/d3sc04317g), [2](https://pubs.acs.org/doi/full/10.1021/jacs.4c17015).

## Usage
Example usage is shown in source/examples/

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

```

## Contributors
Moritz Thürlemann  
Felix Pultar  
Igor Gordiy  

## License

The AMP-BMS code is published and distributed under the [MIT License](LICENSE). 




