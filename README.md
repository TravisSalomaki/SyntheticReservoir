## Travis Salomaki, the University of Texas at Austin | [LinkedIn](https://www.linkedin.com/in/travissalomaki/)

## Motivation

This tool provides a method to generate semi-realistic synthetic reservoir data to support reservoir simulation efforts, spatial modeling, and machine learning workflows. In its current state, the program is able to generate permeability, porosity, depth, and thickness values for any x-y-z combination of grid block geometries. These values are then exported to your working directory as .txt files which can be used in any external modeling program or can be read back into python for further use. 
  
This project was created to support Dr. Matthew Balhoff at the University of Texas at Austin in the creation of a reservoir simulation textbook. This project would not be possible if not for Dr. Michael Pyrcz's geostatistical python library, ["GeostatsPy"](https://github.com/GeostatsGuy/GeostatsPy) which provides the technical backbone through which the simulated data is created. Additionally, portions of the code that create the porosity and permeability values were borrowed from Dr. Michael Pyrcz's workflow titled ["GeostatsPy_synthetic_well_maker"](https://github.com/GeostatsGuy/PythonNumericalDemos/blob/master/GeostatsPy_synthetic_well_maker.ipynb).

## ReadMe
Before running the program, you will most likely have to install GeostatsPy into your anaconda environment if you haven't already. For Mac users this may be tricky as you have to jump through some extra hoops. [Here](https://github.com/GeostatsGuy/GSLIB_MacOS) is a link to further intstructions for installing the necessary executables for Mac users. 

To use the program, an excel file must be created that contains the various inputs of the program. Within the GitHub repository is a template of the input sheet or you can download it directly [here](https://github.com/TravisSalomaki/SyntheticReservoir/raw/main/InputTemplate.xlsx).

The program can be run either in a Jupyter notebook or from a command terminal. Note, interactive plotting is only currently supported within the Jupyter notebook version. 

To run the program, simply instantiate the object and then call its master() method.
~~~~
SyntheticReservoir = SRS()
SyntheticReservoir.master()
~~~~
Running the above code will import all of the necessary inputs from the excel file, generate the synthetic data, and then save the files to your working directory.
If you have multiple rows within your excel file corresponding to different reservoir geometries (i.e nx-ny-nz combinations), the program will generate a unique export file for each row automatically. 

### Visualization Methods

Within the program there are three main visualization tools to let the user inspect various properties of the simulation. 

1. `self.plot_histogram()`
2. `self.plot_layers()`
3. `self.visualize()`

The first method, `self.plot_histogram()` takes two inputs, the data used in the histogram and the desired x-label. The data used in the histogram is either `self.por` or `self.perm`.

The second method, `self.plot_layers()` takes three inputs: the data used to plot, the number of layers to visualize, and the plot title. 

The third method, `self.visualize()`, takes the desired data (e.g `self.por`) as its only input. This method will generate an interactive 3D plot through `%matplotlib qt`. Note, when dealing with large reservoir geometries, this method may take a while to load. 

## Contact

If you encounter any problems with the program or have any general questions about its use feel free to email me at travis.salomaki@gmail.com.

Travis Salomaki, B.S. Petroleum Engineering Honors, the University of Texas at Austin.
