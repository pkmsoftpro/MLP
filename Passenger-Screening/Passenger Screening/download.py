from urllib.request import urlopen
from urllib.request import urlretrieve
url = "https://firebasestorage.googleapis.com/v0/b/passscreen-7dd45.appspot.com/o/images%2Fstage_1.aps?alt=media&token=4725c2e0-698a-4f20-ae85-c4ead747121d"
urlretrieve(url,"stage_1.aps")
# response = urlopen("https://firebasestorage.googleapis.com/v0/b/sample-3035e.appspot.com/o/sweet_gifs%2Fstage_1.aps?alt=media&token=6625bc95-eb65-4f5a-8944-7d91ace6812f")
# html = response.read()
# response = urlopen(url)
# lines = response.readlines()
# content=urllib2.urlopen("https://wordpress.org/plugins/about/readme.txt")
# for line in content:
#     print (line)