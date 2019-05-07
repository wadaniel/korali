korali-plot

## plot_cmaes.py


python plot_cmaes.py resultsdir


- description: loops over all files/generations in resultsdir and plots evolution of cma-es optimizer

- parameter: resultsdir:  relative path to the folder containing all result files

- example1 (from console): python `korali-config --prefix`/bin/plot_cmaes.py korali1/
- example2 (directly from code): ...; korali["Live Plotting"] = true; ...; korali.run();





## plot_tmcmc.py

python plot_tmcmc.py resultsdir

- description: loops over all files in resultsdir and plots samples in histograms
               a single file can be plotted too

- parameter:
    resultsdir: relative path to foler OR single file

- example1 (from console, single file):
    python `korali-config--prefix`/bin/plot_cmaes.py korali1/s00007.json

- example2 (directly from code):
    ...; korali["Live Plotting"] = true; ...; korali.run();


- Comment:

    use `korali-config --prefix` in console to find the installation directory
    of korali, all tools are available in `korali-config--prefix`/bin/
