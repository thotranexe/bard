#BARD- Basic Ai Reccomending Ditties(a short, simple song.)


#Parse User Input to and translate it into readable workable information

#example user would input below:
songs=["Into You","Love And War","Fellas In Paris","Ram Ranch"]

#ideally have a large catalog of music on standby to string match titles
#and grab information like:
genre=["R&B","R&B","RAP","COUNTRY"]
# other things like artist, year released, length, tempo ,etc could be included

#AI Would go here
bias={}
#count the more likely to occure genres
for g in genre:
    if g not in bias:
        bias[g]=1
    else:
        bias[g]+=1
#find most common genre
favorite_genre=max(bias,key=bias.get)

#make reccomendation based on preferences and given information
print("Based on your inputed songs you really like:",favorite_genre)
#draw from catalog to reccomend
rbsongs=["The boy is mine - Monica", "Say my Name - Destiny's Child","Your love - Boyz II Men"]
print("Therefore we recommend:")
for song in rbsongs:
    print(song,'')

# Mostly Machine Learning, Potential to optimize and use supervised learning
# in order to tune the model to better curate songs. Use a large catalog
# of what the user listens to most often to reccomend new songs, of similar
# taste . Perhaps even dip into catergorizing new music itself

#user input -> model -> rec -> user feedback -> reoptimized for preferances -> repeat until desired curation is reached.
# allow hard resets to escape same/ unwanted recomendations.

#or have it like a musical akinator where user is prompted with multiple questions until
#model is 90% sure that it found a song that the user would like.