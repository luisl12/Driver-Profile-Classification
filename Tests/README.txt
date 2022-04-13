The Python script "idreams_pull_trips.py" downloads trips from CardioID's API server.
It uses a token that both authenticates the script to the API and determines which trips the token can access.


Pre-Requisites
--------------

Working Python 3 installation (Python version >= 3.6), with these 3rd party packages:
* pandas
* requests

We recommend installing the Anaconda distribution, which already includes all the needed packages.
https://www.anaconda.com/products/individual


Running the Script
------------------

1. Open a command line shell in the directory where the script is.
   * On Windows, we recommend using the "Anaconda Prompt" shell, which can be opened from the Start menu.
   * Use "cd C:\path\to\directory" to change the directory (on Windows, for example to change to the D:\ drive, type "D:", before using the "cd" command).
2. Type "python idreams_pull_trips.py -ti 2021-04-21 -tf 2021-04-22", replacing for the desired dates.
   * "ti" is the start date of the query period, it is a required parameter.
   * "tf" is the end date of the query period, it is an optional parameter; if not provided, defaults to the current date.
3. Press enter to run the script.
4. Available trips within the query period are downloaded to an automatically created folder "trips", in the same directory of the script.
5. Executing "python idreams_pull_trips.py --help" will print the script's help information.
 