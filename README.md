# mtb-trails
DSI Capstone 2 to find topics in mtb trail descriptions

## Goal

## Data

mtbproject.com has trail data available through their API, that includes the trail name, difficulty, star rating, length, ascent/descent, and a short summary description of the trail.

Snapshot of data on website 

Chose 10 select locations in the US that have a lot of moutain bike trails nearby, based on my previous knowledge and looking at the mtbproject map.  I also tried to choose locations that were spread across different geo regions in the country.

    Portland
    Denver
    Moab
    Pittsburgh
    Los Angeles
    Bellingham WA
    Boise
    Atlanta
    Little Rock
    Grand Rapids


Used the mtbproject API to get the trail data: pulled the maximum of 500 trails within a 100 mile radius of each location.  The API returns json with the following format:

(snapshot)
Some locations appeared to have fewer than 500 trails within this radius, so the resulting dataset was 4365 rows.
Since my project analyzes the summary description of a trail, I removed trails where the summary was akin to "Needs summary.", "Adopt this trail."
Resulting dataset is 4086 rows.

and in df



## EDA

Viewed by state (based on the initial location but some trails within the 100 mile radius may be in a different state.)
By trail difficulty, star rating, length in miles, ascent in feet.
Added a binary column to denote if the trail appears to be a loop or point-to-point; I assumed if the difference between the ascent and descent is less than 1% of the total ascent, then the trail is a loop (otherwise it's a point-to-point).

Word cloud with the most frequent words across all of the trail summaries (after removing stopwords such as 'trail', 'ride', 'area', 'route', 'way', 'feature', 'section', etc.)

Open question whether to remove 'singletrack' as a stop word: it shows up a lot in many topics, but may also convey meaning b/c not all trails are singletrack and may contain sections or the whole trail as road or doubletrack.


## Modeling

LDA using added stop words and spacy lemmatization, 10 topics.  Produced ok topics but not much differentiation.

LDA with only 2 topics, no lemmatization (took forever to run), removing 'singletrack' and 'loop', n-grams=(1,2).  Pretty distinct topics:

    Topic 0:
    ['fun' 'great' 'short' 'fast' 'technical' 'descent' 'downhill' 'views'
    'steep' 'climb']
    Topic 1:
    ['road' 'creek' 'park' 'climb' 'forest' 'lake' 'easy' 'doubletrack'
    'ridge' 'access']

    Model perplexity: 563.720

Topic 0 seems to be more challenging technical trails where the summary describes the ride (technical, fast, steep), and topic 1 looks like easier trails where the summary describes the scenery (creek, views, lake).

3 topics:
    Topic 0:
    ['climb' 'park' 'descent' 'fun' 'steep' 'ridge' 'fast' 'climbs' 'long'
    'scenic']
    Topic 1:
    ['great' 'road' 'creek' 'views' 'short' 'lake' 'nice' 'doubletrack'
    'access' 'connector']
    Topic 2:
    ['fun' 'technical' 'downhill' 'mountain' 'good' 'flowy' 'rock' 'fast'
    'flow' 'rocky']

    Model perplexity: 596.734


## Results

## Next Steps
