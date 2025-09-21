FS Filter-Lab is an optical filter plotting tool, designed to let users plan full-spectrum filter stacks.

It's goal is to make it easier to get into data-driven custom filter stack creation, especially for beginners,
by offering a tool for well-visualized data creation.

Features:
- plot transmission data for camera filters
- show combined transmission curve
- calculate total opacity
- load illuminant and sensor quantum efficincy curve (generic CMOS by default)
- show sensor response curve at given illuminant
- show illuminant and QE graphs independently
- export everything as .png file for easy sharing

IMPORTANT:
You need to grab filter, QE and illuminant data so the program has something to work with. That can be downloaded from here: https://github.com/CheeseCube312/Filter-Plotter-Data

It is no longer directly included in the release to ensure that you can update the software without overwriting your own, manually imported filter library if you want to avoid that. 

_______________________________________________________________

Installation:

1) Install Python 3.8 or higher. Make sure to Install to PATH (should be selectible in install wizard) https://www.python.org/

2) Run Install.bat

Install.bat will download install all the necessary python libraries (found in Requirements.txt)

After first install you can start with start.bat. It just starts the program with a virtual environment (venv)

______________________________________________________________

Adding/Removing data:

Turn graphs into .csv files unsing WebPlotDigitizer: https://apps.automeris.io/wpd4/

Use the Filter Importers in the program to convert them into a usable format.

______________________________________________________________
Filter data format:
It's getting changed with the most recent Refactor, currently still in the Beta/Testing branch. A data conversion script will be added shortly. 
