# Workflow execution

The configuration is currently done in python files contained in the `config` folder.
File paths should be changed or adapted there if the project is ported to a new 
path.

Execute the workflow with the following command:

```
snakemake -pk -c 3 -n
```

Remove the `-n` flag to execute the workflow with a maximum of 3 jobs in parallel. On Windows `--drop-meta` can be added in case the metadata cannot be written.


## Plot setup

Intially we set these parameters for plotting using matplotlib:

```python
import matplotlib.pyplot as plt
import seaborn

plt.rcParams['figure.figsize'] = [4.0, 3.0]
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

plt.rcParams['figure.dpi'] = 147

njab.plotting.set_font_sizes('x-small')

seaborn.set_style("whitegrid")
```

## Rule Graph

```bash
snakemake --forceall --rulegraph | dot -Tpdf > rulegraph.pdf # rule graph
snakemake --forceall --rulegraph | dot -Tpng > rulegraph.png # rule graph
```

![rule grapf](rulegraph.png)