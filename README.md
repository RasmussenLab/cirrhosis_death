# Predict who dies of newly diagnosed cirrhosis patients

This repository contains the code for the analysis presented in the paper:

> Mynster Kronborg, T., Webel, H., O’Connell, M. B., Danielsen, K. V., Hobolth, L., Møller, S., Jensen, R. T., Bendtsen, F., Hansen, T., Rasmussen, S., Juel, H. B., & Kimer, N. (2023).  
> Markers of inflammation predict survival in newly diagnosed cirrhosis: a prospective registry study.  
> Scientific Reports, 13(1), 1–11.  
> https://www.nature.com/articles/s41598-023-47384-2

The core functionaly used is available through the python package njab, which can be installed from PyPI: [pypi.org/project/njab](https://pypi.org/project/njab/)

## Snakemake workflow

The [Snakemake workflow](analysis/Snakefile) builds the data and analyis from the raw data. It was 
executed locally. It allowed to easily re-run the entire pipeline if the data was
updated during the course of the study or if the analysis was updated.

```
cd analysis
snakemake -n 
```

Please see the [analysis README](analysis/README.md) for more details.

> The data is unfortunately not available for download. The repository is meant for 
> documenting the analysis steps and local execution.


## Setup environment


The environment was build using conda

```
# in repo directory
conda env create -f environment.yml
conda activate cirrhosis_death
```

In case you want to inspect the packages in a concrete environment, we provided 
a dump in [environment_231109_win64.yml](environment_231109_win64.yml).