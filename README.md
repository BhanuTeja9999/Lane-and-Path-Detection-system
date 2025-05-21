# Lane-and-Path-Detection-system

1. Uninstall existing versions of Python or Anaconda distribution and its related files in your laptop or PC  
   -> Open Control Panel > Programs > Programs and Features and select the existing version of Python or Anaconda and click on uninstall  

2. Install Python version 3.7.0 from https://www.python.org/downloads/windows/  
   -> Step 2: Run the Installer  
   Once the installer is downloaded, locate it in your Downloads folder and double-click it.  
   Important: On the first screen, check the box "Add Python to PATH" at the bottom.  
   This ensures Python is added to the system environment variables, allowing you to run Python from the command line.  
   Click Install Now  

3. Once the installation is complete, open the Command Prompt (press Win + R, type cmd, and hit Enter).  
   Type the following command and press Enter:  
   `python --version`  
   It will show: `Python 3.7.0`  

4. Install the below libraries in Command Prompt:  
   `pip install pandas==0.25.3`  
   `pip install tensorflow==1.14.0`  
   `pip install matplotlib==3.1.1`  
   `pip install numpy==1.19.2`  
   `pip install scikit-learn==0.22.2.post1`  
   `pip install sklearn-extensions==0.0.2`  
   `pip install keras==2.3.1`  
   `pip install opencv-python==4.1.1.26`  
   `pip install opencv-contrib-python==4.3.0.36`  
   `pip install h5py==2.10.0`  
   `pip install pillow==6.2.1`  
   `pip install PyMySQL==0.9.3`  
   `pip install Django==2.1.7`  
   `pip install pickleshare==0.7.5`  
   `pip install seaborn==0.10.1`  

5. Open the folder of the project using any IDE or navigate to the folder in CMD and type the following command:  
   `py main.py`  

6. A user interface will appear on the screen and click on the "Upload Video" button  
   -> Click on the desired test video  

7. Click on "Lane and Path Detection" button  

8. Output video will appear in a separate window  

9. Hit 'q' to exit  
