# AEGIS

Code for the paper *AEGIS: private generative transformers that learn the natural history of disease, not the
patients*.

AEGIS is a generative transformer of disease histories, built from GPT-2 blocks. It follows the design of Delphi
and is trained with person-level differential privacy (DP-SGD).
This repository contains the method (model, private training, privacy audit and synthetic cohorts), a demo on
random data, and a notebook that runs the whole method on a simulated cohort. It contains no UK Biobank data, no
trained models, no synthetic cohorts and no per-person results.

## System requirements

The code needs Python 3.9 to 3.12 and the packages in `requirements.txt`: PyTorch 2.2.1, Opacus 1.5.2, NumPy
(below 2.0), SciPy, pandas, Matplotlib and JupyterLab. The paper's analyses ran on Linux with Python 3.10 and one GPU.
The code is plain Python and should also run on macOS and Windows, but we tested it on Linux
only. The demo needs no GPU; the notebook runs on a CPU at a smaller size.

## Installation

Clone the repository and install the requirements in a virtual environment:

```
git clone https://github.com/<username>/AEGIS.git
cd AEGIS
python -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Installation takes a few minutes. Run everything from the main folder of the repository.

## Training

To check the installation, run the demo:

```
python demo.py
```

It trains small models on random data with and without DP-SGD, then runs the privacy audit and sampling. It should
end with `PASSED: 22 of 22 checks passed` after 1 to 5 minutes on a CPU.

To train AEGIS on a simulated cohort, open the notebook and run all cells:

```
jupyter lab aegis_walkthrough.ipynb
```

It trains one model without privacy and one with DP-SGD at epsilon 8, with the paper's architecture and DP-SGD
settings, then evaluates, audits and samples them. A full run takes about 100 minutes on one GPU; without a GPU it
uses a smaller cohort and model and takes about an hour on a CPU. The numbers come from a simulation and are not the
paper's results.

## Using AEGIS on other data

Disease histories go into `aegis.data.Cohort`: one block of records per person, each record a token of the fixed
vocabulary (`aegis.data.Vocab`) and an age in days. `simulation.py` shows how to build a `Cohort`, and the setup
section of the notebook shows the rest of the calls in order: `make_splits` to split by person,
`aegis.train.train` to train with or without DP-SGD, `aegis.audit.run_audit` for the privacy audit, and
`aegis.synthetic.ModelGenerator` to sample synthetic people. The public functions are documented in their
docstrings.

## Data availability

The paper uses UK Biobank data, which are available to approved researchers through an application to UK Biobank
(https://www.ukbiobank.ac.uk). We cannot share these data, the models trained on them or any per-person result. The
aggregate results behind the paper's figures are in its Supplementary Data 1, with every count of 1 to 9 people
suppressed. The demo and the notebook generate their own data, so nothing needs to be downloaded.

## Notes on privacy

The planted patients of the paper are drawn from the rarest codes of the training members (`planted_pool` with
`kind='rare'`). This choice depends on the training data, so the DP guarantee of the models trained with these
patients holds for a fixed planted set, and these models and their synthetic cohorts are audit models (see the
paper's Methods). A model meant for release is trained without planted patients, or with `kind='any'`, which
draws the codes from the public vocabulary.

Opacus was used with `secure_mode=False`, as in most research use; a model meant for release would be trained with
a secure random number generator. The same holds for the Laplace noise of `dp_real_rate` (rate calibration), which
is drawn from a seeded generator so that the audit runs reproduce.

## Citation

Coming soon.

## Licence


This code is released under the MIT licence; see `LICENSE`. AEGIS adapts parts of the Delphi code (Shmatko et al., Nature 2025), which is also released under the MIT licence; see `THIRD_PARTY_NOTICES.md`.
