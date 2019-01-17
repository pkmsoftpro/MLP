# Passenger-Screening
We are designing an algorithm in Python using OpenCV, Alexnet, and Tensorflow to read passenger images of file types aps, a3d, a3das, and ahi, and to detect possible threats that these images may show.  Machine learning plays a key role in figuring out whether the person in the images used has an object on them or not.  

# The Algorithm
The algorithm is divided up into at least three parts: Preprocessing/Training, Scanning, and Results.

Preprocessing/Training

Image files (like the .aps files provided from Kaggle) are used to train the machine to see images of passengers with and without potential threats.  Each image is divided up into 17 zones.  Each image file has 16 images of a passenger at different angles.  Each zone is visible for less than 16 images, so only the images that a specific zone appears in are used for that zone.  Each zone's data was saved in an array, and each array contains all passenger images in that specific zone.  When training the machine, a majority bias occurred.  This bias caused inaccurate results.  If one set of images has a threat in it, but the rest do not, the machine would assume that there no threat.  In order to significantly decrease the chances of this bias, only three random images of each passenger with no threat on them were saved, while all images of each passenger with a threat on them were saved.


Scanning

After training the machine, it could be used as a scanner to detect if an .aps file's images include a threat in one or more specific zones.


Results

The results are shown, which include loss and accuracy of each threat zone.  The average accuracy is about 90.58%.


# Front End Work
The front end of this project involved creating a web app to upload images, download images, and show results.

Passenger_Screening_Interface.html

This html file is used to upload images to the PassScreen project on Firebase.  These images are sent to storage for later use.  Public access needs to be allowed in order for these images to be uploaded.


Download.py

This python file downloads an image file that had been uploaded to the project on Firebase.  The url of the image file is used in Download.py in order for this to be fulfilled.  To download, use an Anaconda prompt or any terminal and type: python Download.py (make sure the terminal is in the correct directory).  The image file should appear right in the same directory that Download.py is in.


output_html.py

This python file creates an html file called Output.html.  The html file was going to be designed to show the results from scanning a .aps file's images, such as whether there is a threat or no threat, and if there is a threat, which zone(s).  The file only shows an example of how it could have displayed these results.


An updated version of the webpage is in the Passenger Screening file.  This webpage has features, such as our names and the ability to contact us.
