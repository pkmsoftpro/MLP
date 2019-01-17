# Download the stage_1.aps file from the Firebase Storage of the passenger screening project.
from urllib.request import urlopen
from urllib.request import urlretrieve
url = "https://firebasestorage.googleapis.com/v0/b/passscreen-7dd45.appspot.com/o/images%2Fstage_1.aps?alt=media&token=4725c2e0-698a-4f20-ae85-c4ead747121d"
urlretrieve(url,"stage_1.aps")
