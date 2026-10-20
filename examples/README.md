# Examples

| Notebook | What it covers |
| --- | --- |
| [`dustpedia_dust_scaling.ipynb`](dustpedia_dust_scaling.ipynb) | Dust scaling relations of 229 nearby galaxies from DustPedia, built up from a 2-D fit without uncertainties to a 5-D fit with uncertainties, HI upper limits and missing HI data. It also covers significance testing with posterior probabilities, Savage-Dickey Bayes factors and the parametric-bootstrap p-value, and partial correlations. |

## Running the notebooks

With [pixi](https://pixi.sh), from the repository root:

```
pixi run -e examples examples        # open JupyterLab in this directory
pixi run -e examples run-examples    # re-execute every notebook in place
```

The `examples` environment includes JupyterLab and the `nutpie` sampler. Without pixi,
`pip install "pybaycor[nutpie]" jupyterlab` is enough. The notebooks read their data
relative to this directory, so start Jupyter here.

## Data

`data/dustpedia_sample.csv` combines two public DustPedia catalogues from CDS:

- Nersesian et al. (2019, A&A 624, A80): CIGALE/THEMIS fits that give stellar masses,
  star-formation rates, dust masses and dust temperatures;
- De Vis et al. (2019, A&A 623, A5): HI masses and non-detection flags.

`data/build_dustpedia_sample.py` downloads the original tables and regenerates the
file. Please cite these papers, and Davies et al. (2017, PASP 129, 044102) for
DustPedia, if you use the data.
