# korali-plot

##Name
       korali-plot - Creates a graphical representation and plots for the results produced
	   by a Korali application.
	   
##Usage
       korali-plot [RESULT_FOLDER] [-o OUTPUT_FORMAT] [-f OUTPUT_FILE]
				   
##Description

       Given the input folder, korali-plot determines which solver was used by the
	   Korali application and calls the appropriate Korali plotting script:
	   
	   - For TMCMC: plot_tmcmc.py
	   - For CMA-ES: plot_cmaes.py
	   
	   Although these scripts can be manually instantiated by the user, korali-plot
	   provides the additional convenience of automatically determining which solver
	   was used.

       
##Options

      RESULT_FOLDER
        Specifies the input folder containing Korali results.
		Default value: ./_Korali_results
		
	  OUTPUT_FORMAT
        Specifies the format of the produced plots. Possible values are:
		   -o display (default)
              Shows the live results as a new window through the graphical interface.
		   -o png
		      Creates a PNG file for each generation.
		   -o mp4 
		      Creates a MP4 video animation containing all generations.
			  
	  OUTPUT_FILE
        Specifies the file name root for output plotting. Korali will automatically additional
		a generation number as suffix and the corresponding extension (e.g., .png). Example
		   -f myOutputFile
	   
		
