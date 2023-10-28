# Import the following packages
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import math

# Download the following packages if not already downloaded
# nltk.download("punkt")
# nltk.download("wordnet")
# nltk.download("omw-1.4")

# Create a tokenizer object
tokenizer = RegexpTokenizer(r'\w+')

# Create a stemmer and lemmatizer object
ps = PorterStemmer()
wnl = WordNetLemmatizer()

# Create a list of documents
documents = [
    "Chandrayaan, series of Indian lunar space probes. Chandrayaan-1 (chandrayaan is Hindi for “moon craft”), the first lunar space probe of the Indian Space Research Organisation (ISRO), found water on the Moon. It mapped the Moon in infrared, visible, and X-ray light from lunar orbit and used reflected radiation to prospect for various elements, minerals, and ice. It operated in 2008-09. Chandrayaan-2, which launched in 2019, was designed to be ISRO's first lunar lander. Chandrayaan-3 was ISRO's first lunar lander and touched down in the Moon's south polar region in 2023.",
    "A Polar Satellite Launch Vehicle launched the 590-kg (1,300-pound) Chandrayaan-1 on October 22, 2008, from the Satish Dhawan Space Centre on Sriharikota Island, Andhra Pradesh state. The probe then was boosted into an elliptical polar orbit around the Moon, 504 km (312 miles) high at its closest to the lunar surface and 7,502 km (4,651 miles) at its farthest. After checkout, it descended to a 100-km (60-mile) orbit. On November 14, 2008, Chandrayaan-1 launched a small craft, the Moon Impact Probe (MIP), that was designed to test systems for future landings and study the thin lunar atmosphere before crashing on the Moon's surface. MIP impacted near the south pole, but, before it crashed, it discovered small amounts of water in the Moon's atmosphere.",
    "The U.S. National Aeronautics and Space Administration (NASA) contributed two instruments, the Moon Mineralogy Mapper (M3) and the Miniature Synthetic Aperture Radar (Mini-SAR), which sought ice at the poles. M3studied the lunar surface in wavelengths from the visible to the infrared in order to isolate signatures of different minerals on the surface. It found small amounts of water and hydroxyl radicals on the Moon's surface. M3also discovered in a crater near the Moon's equator evidence for water coming from beneath the surface. Mini-SAR broadcast polarized radio waves at the north and south polar regions. Changes in the polarization of the echo measured the dielectric constant and porosity, which are related to the presence of water ice. The European Space Agency (ESA) had two other experiments, an infrared spectrometer and a solar wind monitor. The Bulgarian Aerospace Agency provided a radiation monitor.",
    "The principal instruments from ISRO—the Terrain Mapping Camera, the HyperSpectral Imager, and the Lunar Laser Ranging Instrument—produced images of the lunar surface with high spectral and spatial resolution, including stereo images with a 5-metre (16-foot) resolution and global topographic maps with a resolution of 10 metres (33 feet). The Chandrayaan Imaging X-ray Spectrometer, developed by ISRO and ESA, was designed to detect magnesium, aluminum, silicon, calcium, titanium, and iron by the X-rays they emit when exposed to solar flares. This was done in part with the Solar X-Ray Monitor, which measured incoming solar radiation.",
    "Chandrayaan-1 operations were originally planned to last two years, but the mission ended on August 28, 2009, when radio contact was lost with the spacecraft.Chandrayaan-2 launched on July 22, 2019, from Sriharikota on a Geosynchronous Satellite Launch Vehicle Mark III. The spacecraft consisted of an orbiter, a lander, and a rover. The orbiter circles the Moon in a polar orbit at a height of 100 km (62 miles) and has a planned mission lifetime of seven and a half years. The mission's Vikram lander (named after ISRO founder Vikram Sarabhai) was planned to land on September 7. Vikram carried the small (27-kg [60-pound]) Pragyan (Sanskrit: “Wisdom”) rover. Both Vikram and Pragyan were designed to operate for 1 lunar day (14 Earth days). However, just before Vikram was to touch down on the Moon, contact was lost at an altitude of 2 km (1.2 miles).",
    "Chandrayaan-3 launched from Sriharikota on July 14, 2023. The spacecraft consists of a Vikram lander and a Pragyan rover. The Vikram lander touched down on the Moon on August 23. It became the first spacecraft to land in the Moon's south polar region where water ice could be found under the surface. The landing site was the farthest south that any lunar probe had touched down, and India was the fourth country to have landed a spacecraft on the Moon—after the United States, Russia, and China.",
]

# Find length of the documents
N = len(documents)

# Create a query
query = "Chandrayaan 3 had a Vikram lander and Rover"


'''Preprocessing of the documents and query'''
# Tokenize the documents and query
newDocuments = []
for doc in documents:
    newDocuments.append(tokenizer.tokenize(doc))

newQuery = tokenizer.tokenize(query)

# Transform the tokens of the documnets and the query to lower case, lemmatize and stem the tokens
for i in range(N):
    for j in range(len(newDocuments[i])):
        newDocuments[i][j] = newDocuments[i][j].lower()
        newDocuments[i][j] = wnl.lemmatize(newDocuments[i][j], pos="v")
        newDocuments[i][j] = ps.stem(newDocuments[i][j])

for j in range(len(newQuery)):
    newQuery[j] = newQuery[j].lower()
    newQuery[j] = wnl.lemmatize(newQuery[j], pos="v")
    newQuery[j] = ps.stem(newQuery[j])

# Make a set of all the words in the documents
wordSet = set()
for doc in newDocuments:
    for word in doc:
        wordSet.add(word)

# Initialize the dictionaries
wordDict = []
TFDict = []
weights = []
queryDict = dict.fromkeys(wordSet, 0)
queryTFDict = dict.fromkeys(wordSet, 0)
queryWeights = dict.fromkeys(wordSet, 0)

for i in range(N):
    wordDict.append(dict.fromkeys(wordSet, 0))
    TFDict.append(dict.fromkeys(wordSet, 0))
    weights.append(dict.fromkeys(wordSet, 0))

# Find the frequency of each word in each document
for i in range(N):
    for word in newDocuments[i]:
        wordDict[i][word] += 1

# Find the frequency of each word in the query
for word in newQuery:
    queryDict[word] += 1

# Find the Term Frequecny (TF) of each word in each document
for i in range(N):
    for word in wordDict[i]:
        if wordDict[i][word] > 0:
            TFDict[i][word] = 1 + math.log2(wordDict[i][word])

# Find the Term Frequecny (TF) of each word in the query
for word in queryDict:
    if queryDict[word] > 0:
        queryTFDict[word] = 1 + math.log2(queryDict[word])

# Find the Inverse Document Frequency (IDF) of each word
IDFDict = dict.fromkeys(wordSet, 0)
for word in IDFDict:
    count = 0
    for i in range(N):
        if word in newDocuments[i]:
            count+=1
        if count!=0:
            IDFDict[word] = math.log2(N/count)

# Find the weights of each word in each document
for i in range(N):
    for word in TFDict[i]:
        weights[i][word] = TFDict[i][word]*IDFDict[word]

# Find the weights of each word in the query
for word in queryTFDict:
    queryWeights[word] = queryTFDict[word]*IDFDict[word]

# Make a list of the weights of each document
arrayOfWeights = []
for i in range(N):
    arrayOfWeights.append(list(weights[i].values()))

arrayOfQueryWeights = list(queryWeights.values())

# Find the squared sum of the weights of each document
squaredSum = [0]*N
for i in range(N):
    for j in range(len(arrayOfWeights[i])):
        squaredSum[i]+=(arrayOfWeights[i][j]*arrayOfWeights[i][j])

stopCriterion = 3
ThresholdClass = 0.5

# Clustering using complete link algorithm
while N>stopCriterion:
    sim = [ [0] * N for i1 in range(N) ]
    maxi = -1
    for i in range(N):
        for j in range(N):
            dotProduct = 0
            for k in range(len(arrayOfWeights[i])):
                dotProduct += arrayOfWeights[i][k]*arrayOfWeights[j][k]
            sim[i][j] = dotProduct/(math.sqrt(squaredSum[i])*math.sqrt(squaredSum[j]))
            if i!=j and sim[i][j]>maxi:
                maxi = sim[i][j]
                maxIndex = [i, j]
    print("The most similar documents are: ", maxIndex[0]+1, " and ", maxIndex[1]+1, " with similarity: ", maxi)
    print("Merge documents using complete link algorithm", maxIndex[0]+1, " and ", maxIndex[1]+1)
    mergedDoc = [max(arrayOfWeights[maxIndex[0]][k], arrayOfWeights[maxIndex[1]][k]) for k in range(len(arrayOfWeights[0]))]
    # Remove the two merged documents from the list of documents
    arrayOfWeights.pop(maxIndex[1])
    arrayOfWeights.pop(maxIndex[0])
    # Add the merged document to the list of documents
    arrayOfWeights.append(mergedDoc)
    # Update the squared sum of the weights
    squaredSum.pop(maxIndex[1])
    squaredSum.pop(maxIndex[0])
    squaredSum.append(sum([w**2 for w in mergedDoc]))
    # Update the number of documents
    N -= 1

# Similarity between query and documents in final clusters
maxIndex = -1
for i in range(N):
    maxi = -1
    dotProduct = 0
    for k in range(len(arrayOfWeights[i])):
        dotProduct += arrayOfWeights[i][k]*arrayOfQueryWeights[k]
    sim = dotProduct/(math.sqrt(squaredSum[i])*math.sqrt(sum([w**2 for w in arrayOfQueryWeights])))

    print("Similarity between query and cluster ", i+1, " is: ", sim)
    if sim>maxi:
        maxi = sim
        maxIndex = i

print("The most similar cluster is: ", maxIndex+1)

# Get the words in the most similar cluster as a dict with word as key and weight as value
clusterDict = dict.fromkeys(wordSet, 0)
for word in clusterDict:
    clusterDict[word] = arrayOfWeights[maxIndex][list(wordSet).index(word)]

# Sort the clusterDict in descending order of weights
sortedClusterDict = {k: v for k, v in sorted(clusterDict.items(), key=lambda item: item[1], reverse=True)}

# Get the top 5 words from the sortedClusterDict which are not in the query
top5 = []
for word in sortedClusterDict:
    if word not in newQuery:
        top5.append(word)
    if len(top5)==5:
        break
print(top5)

# Append the top 5 words to the query as a string
for word in top5:
    query += " " + word
print("The new query is: ", query)




