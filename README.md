# higrid
## Create a 3-D image of galactic hydrogen observations
### Glen Langston - 2026 March 18

Create a 3-D cube with a variety of options for how the spectra are gridded
for different science goals.   This is the first working version of this program.

## Usage
To get help type:
python higrid.py

## Usage Example:

 python higrid.py  --projection AIT --baseline --hot /home/karl/keep/I1-26-01-25*.hot --cold /home/karl/keep/I1-26-01-24T*.ast /media/karl/pi1-data-26Mar12/*T*.ast

Where the data files are in "Aficionado" format ascii spectra (.ast).  The calibraiton is accomplished by comparision of the measurements of "cold" sky and "hot" ground.   The data files to be gridded have extensions of 
".ast".   By default, an all sky image is produced.
