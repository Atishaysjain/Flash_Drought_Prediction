# Flash_Drought_Prediction

## Getting Started
- Clone this repository
   ```Shell
   git clone git@github.com:Atishaysjain/Flash_Drought_Prediction.git;
   ```
   
Navigate to the Flash_Drought_Prediction folder 

- Set up a Python3 environment with the provided requirements.txt file.
   ```Shell
   # This environment should have all packages required to execute the code
   pip install -r requirements.txt
   ```
   
- Create two folders
   ```Shell
   # This environment should have all packages required to execute the code
   mkdir Results # Will store the flash drought prediction results
   mkdir Data # Will contain the raw data to be operated upon
   ```
 

## Running the code

```Shell
   !python main.py --num
   ```
 The value of "num" will be equivalent to the value of the last entry of the last row of Lstm_Results.txt. If running the code for the first time, then the value of num should be 0.
