# Results

## Plotting Results

The Korali plotter creates a graphical representation and plots for the results produced by a Korali application.

	python3 -m korali.plotter [RESULT_FOLDER] [-o OUTPUT_FORMAT] [-f OUTPUT_FILE]
		    
Given the input folder, korali-plot determines which solver was used by the
Korali application and calls the appropriate Korali plotting script:
 
*RESULT_FOLDER*

	Specifies the input folder containing Korali results.
	Default value: ./_korali_result

*OUTPUT_FORMAT*

	Specifies the format of the produced plots. Possible values are:
	   -o display (default)
		  Shows the live results as a new window through the graphical interface.
	   -o png
		  Creates a PNG file for each generation.
	   -o mp4 
		  Creates a MP4 video animation containing all generations.
		  
*OUTPUT_FILE*

	Specifies the file name root for output plotting. Korali will automatically additional
	a generation number as suffix and the corresponding extension (e.g., .png). Example
	   -f myOutputFile
	   
## Accessing Results

### Within your Korali Application

!!!warning
	Daniel: Can you explain how and write an example as to how to access results here?

### Exporting Results to Other Formats

!!!warning
	Daniel / George: How about developing one such tool?

The Korali export tool converts Korali's JSON-based result information into a format readable by other applications (e.g., MATLAB).

	python3 -m korali.export [PROPERTY] [RESULT_FOLDER] [-o OUTPUT_FORMAT]
		    
*PROPERTY*

	The result property to export.
	
*RESULT_FOLDER*

	Specifies the input folder containing Korali results.
	Default value: ./_korali_result

*OUTPUT_FORMAT*

	Specifies the output format. Possible values are:
	   -o csv (default)
		  Creates comma-separated value file.

*OUTPUT_FILE*

	Specifies the file name root for output.
	   -f myOutputFile
	   
## Resume Execution

Any saved state produced by Korali can be taken as checkpoint and resume its execution.

### Example: Loading and Changing Configuration.

To load only the configuration of a saved state, we provide the **loadConfig** operation.

```python
#Loading the initial configuration from a Korali saved state.
k.loadConfig('_korali_result/s00005.json')

#We still need to define the model, which is not saved.
k.setModel(myModel)

#Changing its conduit to run un parallel
k[Conduit] = "Distributed"
k["Distributed"]["Ranks Per Team"] = 16

k.run()
```

### Example: Continuing Execution

To load the entire state of a saved state, we provide the **loadState** operation.

```python
#Loading the state from a Korali saved state.
k.loadState('_korali_result/s00005.json')

#We still need to define the model, which is not saved.
k.setModel(myModel)

#Continue Running
k.run()
```
