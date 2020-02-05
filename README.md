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


Used the mtbproject API to get the trail data: pulled the maximum of 100 trails within a 100 mile radius of each location (~1000 total trails).  The API returns json with the following format:

(snapshot)
Some locations appeared to have fewer than 500 trails within this radius, so the resulting dataset was 4365 rows.
Since my project analyzes the summary description of a trail, I removed trails where the summary was akin to "Needs summary.", "Adopt this trail."
Resulting dataset is 4086 rows.

and in df

Turns out mtb project trail descriptions are very short, about one short sentence.


## EDA

Viewed by state (based on the initial location but some trails within the 100 mile radius may be in a different state.)
By trail difficulty, star rating, length in miles, ascent in feet.
Added a binary column to denote if the trail appears to be a loop or point-to-point; I assumed if the difference between the ascent and descent is less than 1% of the total ascent, then the trail is a loop (otherwise it's a point-to-point).

Word cloud with the most frequent words across all of the trail summaries (after removing stopwords such as 'trail', 'ride', 'area', 'route', 'way', 'feature', 'section', etc.)

Open question whether to remove 'singletrack' as a stop word: it shows up a lot in many topics, but may also convey meaning b/c not all trails are singletrack and may contain sections or the whole trail as road or doubletrack.

## Featuratization

stop words
bigram phraser
lemmatize
more stop words

TF params: max_df, min_df, max_features
sklearn LDA vs. gensim LDA

LDA params: # topics, alpha, eta

Perplexity, coherence



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


Singletracks descriptions LDA, 10 topics, not lemmatized, no n-grams.

    Topic 0:
    ['river' 'good' 'narrow' 'hikers' 'traffic' 'spots' 'foot' 'worth'
    'expect' 'run']
    Topic 1:
    ['lake' 'climbs' 'good' 'peak' 'views' 'technical' 'fun' 'areas' 'camp'
    'great']
    Topic 2:
    ['mountain' 'jumps' 'features' 'big' 'advanced' 'hills' 'beginner' 'built'
    'open' 'large']
    Topic 3:
    ['rd' 'springs' 'nice' 'hot' 'mi' 'crosses' 'great' 'pine' 'killpecker'
    'fr']
    Topic 4:
    ['mountain' 'great' 'map' 'technical' 'different' 'sure' 'network' 'bull'
    'lots' 'pretty']
    Topic 5:
    ['valley' 'slickrock' 'sand' 'scenery' 'plenty' 'rim' 'moab' 'great'
    'sweet' 'rough']
    Topic 6:
    ['little' 'rocks' 'fun' 'climbs' 'river' 'technical' 'direction' 'red'
    'good' 'easy']
    Topic 7:
    ['park' 'lake' 'hiking' 'state' 'mountain' 'biking' 'areas' 'open'
    'forest' 'fs']
    Topic 8:
    ['track' 'single' 'great' 'roads' 'place' 'double' 'technical' 'loops'
    'skills' 'wide']
    Topic 9:
    ['road' 'creek' 'climb' 'follow' 'downhill' 'steep' 'ridge' 'canyon'
    'head' 'end']
    Model perplexity: 806.284


Singletracks LDA with gensim -- spread out, but topics don't make sense.



Singletrack NMF with sklearn: 6 topics, stopwords, 2-grams, no lemmatization:

    Topic 0:
    ['ridge' 'steep' 'river' 'canyon' 'end' 'old' 'll' 'climbs' 'rd' 'gravel']
    Topic 1:
    ['creek' 'crossings' 'creek crossings' 'fisher' 'bear creek' 'lower'
    'bear' 'gulch' 'steep' 'portion']
    Topic 2:
    ['provincial' 'hiking' 'swimming' 'fishing' 'lake' 'acres' 'acre'
    'camping' 'activities' 'permitted']
    Topic 3:
    ['beginner' 'intermediate' 'features' 'technical' 'advanced' 'loops'
    'open' 'jumps' 'country' 'hills']
    Topic 4:
    ['lots' 'roots' 'rocks' 'climbs' 'pretty' 'technical' 'challenging'
    'network' 'tough' 'lake']
    Topic 5:
    ['fun' 'little' 'bit' 'little bit' 'easy' 'hard' 'people' 'quick' 'built'
    'pretty']


Top Trails in each topic:

    Topic 0:
    Golden Gate Canyon State Park: Colorado
    Coopers Rock: West Virginia
    West Bench Trail: Colorado
    Bear Creek Lake Park: Colorado
    Hurkey Creek: California
    Troy Built Trail: Colorado
    Hudson Springs Park: Ohio
    Kokopelli Trail - Loma To Moab: Colorado
    Buckhorn Loop: California
    Kipp's Loop: Colorado


    Topic 1:
    West Face: Idaho
    Deadwood Ridge Trail: Idaho
    Orchard / 5 Mile / Watchman: Idaho
    Blankets Creek: Georgia
    Bald Mountain Bike Park: Idaho
    Timberline to Town: Oregon
    Lower Hulls Gulch: Idaho
    Ada/Eagle Bike Park: Idaho
    Fourth Of July Lake / Ants Basin / Robinson Bar: Idaho
    Saddle / Trail #326: Idaho

    Topic 2:
    Brandywine Falls Provincial Park: British Columbia
    Team Pangor: British Columbia
    Alice Lake Provincial Park: British Columbia
    Test Of Metal: British Columbia
    nan: nan
    Gibbs Lake County Park: Washington
    Severed Dick: British Columbia
    Little Mountain : Washington
    Arlington airport downhill trail: Washington
    Heart Lake - Acfl: Washington


    Topic 3:
    Allegan State Game Area: Michigan
    Carbonate Training Loop: Idaho
    Pigeon Mountain: Georgia
    Merrell Trail: Michigan
    Pleasant Valley: Pennsylvania
    Lightning Bend: Michigan
    Fort Ebey St Park: Washington
    Daisy State Park: Arkansas
    Bartlett Wash: Utah
    Eagle Rock Loop: Arkansas


    Topic 4:
    Turnbull Canyon: California
    Blue Jay: Arkansas
    Pando Ski Area: Michigan
    Grand Vue Mountain Bike Trail: West Virginia
    West Branch: Ohio
    Falls Creek: Washington
    Pine Mountain Recreation Area: Georgia
    Luke's Trail: Utah
    Waterworks Blue: Arkansas
    Rattlesnake Mtn: Washington


    Topic 5:
    Georgia International Horse Park: Georgia
    Dauset Trails Nature Center: Georgia
    Thomas City Park and Trails: West Virginia
    Sandy Ridge Trail: Pennsylvania
    Lost Lake Tr/loop: Washington
    Round Lake: Washington
    Mount Seymour: British Columbia
    Pumphouse: Arkansas
    Heli Pad Shortcut: California
    Kokopelli Area Trails: Colorado


## Results

## Next Steps
