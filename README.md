# sensor-forecast
forecasting sensor data with vector autoregression 


Some tasks for the future:
```
[ ] account for moving forecast window 
[ ] try some alternatives for correlation
[ ] take into account the soil moisture external changes (watering), so maybe it does not have to be excluded
[ ] add other groups to compare not only the 3 suggested, and explain the choice
[ ] add and include other attributes as input to model training

[ ] add LSTM network extension
```


## Run in virtual environment to fix issues with imported libraries:
Open a terminal (or command prompt in Windows).
Navigate to the directory where you want to create the virtual environment, for example:
```
cd /path/to/your/project
```
Run the following command to create a virtual environment named "myenv" (you can replace "myenv" with any name you prefer):

```
python -m venv myenv
```
Activate the virtual environment:
For Windows:
```
myenv\Scripts\activate.bat
```
Now, you should see the virtual environment's name in your terminal prompt. This means the virtual environment is active, and you can install packages and run your Python script within this isolated environment.

Install required packages:
```
pip install pandas scikit-learn
pip install keras
pip install tensorflow
```
Run your Python script:
```
python your_script.py
```
When you're done working in the virtual environment, 
you can deactivate it:
```
deactivate
```

The terminal prompt will return to its normal state, 
indicating that you're no longer in the virtual environment.



