---
layout: post
title: Simple Word2Vec using SVD
date: 2018-06-14
description: # add
type_post: same
url_post: # add
img:  # Add image post (optional)
tags: [SVD,Word2Vec]
---
In this i am discussing about creating word vectors using SVD and co-occurrence matrix. For doing this used data base of amazon 
fine food reviews data set, downloded from [kaggle](https://www.kaggle.com/snap/amazon-fine-food-reviews/data). Cleaned the data
and after that took top 5000 top tfidf words because of time taken to calculate co-occurrence matrix is high. Created co-occurrence matrix
as below
~~~ python
def cooccurrence_matrix(list_words,distance,sentances):
    '''
    Returns co-occurrence matrix of words with in a distance of occurrrence
    input:
    list_words: list of words to get the co-occurrance matrix in order
    distance: distance between words
    sentances: documets to check ( a list )
    output:
    co-occurance matrix in te order of list_words order
    '''
    #length of matrix needed
    l = len(list_words)
    #creating a zero matrix
    com = np.zeros((l,l))
    #creating word and index dict
    dict_idx = {v:i for i,v in enumerate(list_words)}
    for sentence in sentances:
        sentence=sentence.strip()
        tokens= sentence.split()
        for pos,token in enumerate(tokens):
            #if eord is in required words
            if token in list_words:
                #start index to check any other word occure or not
                start=max(0,pos-distance)
                #end index
                end=min(len(tokens),pos+distance+1)
                for pos2 in range(start,end):
                    #if same position
                    if pos2==pos:
                        continue
                    # if same word
                    if token == tokens[pos2]:
                        continue
                    #if word found is in required words
                    if tokens[pos2] in list_words:
                        #index of word parent
                        row = dict_idx[token]
                        #index of occurance word
                        col = dict_idx[tokens[pos2]]
                        #adding value to that index
                        com[row,col] = com[row,col] + 1
    return com
~~~
So co-occurrence matrix is (n x n) matrix where n is no of words. it gives how frequent those words exists in a given distance. performed Tuncated SVD on the co-occurrence matrix and selected no of components/dimensions using variance explained and for my data selected as 200 dimensions. Now i have 200 dimension vector for each word.  and converted into dict as below.
~~~ python
#word 2 vectors dict
W2V_200d = {}
j = 0
for i in Words:
    W2V_200d[i] = svd_out[j]
    j = j + 1
~~~
We have 200 dimension vectors for each word and checked similary of vectors using cosine similarity like below.
~~~ python
def top_similar_words(v,Wordvectors,n,list_words):
	"""
	Gives output angles between one vector(v1) and all other vectors as matrix(v2)
	input:
	v shape (d,1)
	Wordvectors shape (m,d)
	n - no of similarities needed
	list_words - list of all words in Wordvectors in same order
	output:
	list of n words which are similar
	"""
	v1 = np.array(v)
	v1 = v1.reshape(-1,1)
	v2 = np.array(Wordvectors)
	#dot product
	nr = np.dot(v2,v1)
	#vector lengths 2-norm
	dr = (np.linalg.norm(v1)*np.linalg.norm(v2,axis=1)).reshape(-1,1)
	# angle i.e cosine inverse and domine for cosine inverse is
	# [-1,1] so clipping to [-1,1] and claculating angle
	ang = np.arccos(np.clip(nr/dr,-1,1))
	ang = ang.ravel()
	#sorting and getting index of top n
	ang_10 = np.argsort(ang)[0:n+1]
	ang_n = np.delete(ang_10,0)
	dict_tfidf = {i:v for i,v in enumerate(list_words)}
	return [dict_tfidf[i] for i in ang_n]
~~~
found similar words like below
For 'Good' got top 10 as ('enjoy', 'enjoying', 'fantastic', 'amazing', 'fabulous', 'nice', 'delicious', 'incredible', 'outstanding', 'perk')  
For 'sep' - ('dec', 'aug', 'august', 'november', 'sept', 'february', 'expired', 'expiry', 'exp', 'expiration')  
For 'yum' - ('yummy', 'delish', 'awesome', 'org', 'everybody', 'addictive', 'wow', 'amazing', 'addicted')  
For 'gin' - ('cocktail', 'cola', 'fizzy', 'clot', 'kinda', 'ting', 'cytoma', 'sour', 'rootbeer', 'soda') 
  
References:
1. applied ai course - Matrix factorization methods


<head>    
    <meta http-equiv="content-type" content="text/html; charset=UTF-8" />
    <script>L_PREFER_CANVAS = false; L_NO_TOUCH = false; L_DISABLE_3D = false;</script>
    <script src="https://cdn.jsdelivr.net/npm/leaflet@1.2.0/dist/leaflet.js"></script>
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/1.11.1/jquery.min.js"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.2.0/js/bootstrap.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.js"></script>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.2.0/dist/leaflet.css"/>
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.2.0/css/bootstrap.min.css"/>
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.2.0/css/bootstrap-theme.min.css"/>
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/font-awesome/4.6.3/css/font-awesome.min.css"/>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.css"/>
    <link rel="stylesheet" href="https://rawgit.com/python-visualization/folium/master/folium/templates/leaflet.awesome.rotate.css"/>
    <style>html, body {width: 100%;height: 100%;margin: 0;padding: 0;}</style>
    <style>#map {position:absolute;top:0;bottom:0;right:0;left:0;}</style>
    
            <style> #map_5d87149c73ac45bb9f7cba04c6707063 {
                position : relative;
                width : 100.0%;
                height: 100.0%;
                left: 0.0%;
                top: 0.0%;
                }
            </style>
        
</head>
<body>    
    
            <div class="folium-map" id="map_5d87149c73ac45bb9f7cba04c6707063" ></div>
        
</body>
<script>    
    

            
                var bounds = null;
            

            var map_5d87149c73ac45bb9f7cba04c6707063 = L.map(
                                  'map_5d87149c73ac45bb9f7cba04c6707063',
                                  {center: [40.734695,-73.990372],
                                  zoom: 10,
                                  maxBounds: bounds,
                                  layers: [],
                                  worldCopyJump: false,
                                  crs: L.CRS.EPSG3857
                                 });
            
        
    
            var tile_layer_09b05eec45eb469fa67fe23e9fc91595 = L.tileLayer(
                'https://stamen-tiles-{s}.a.ssl.fastly.net/toner/{z}/{x}/{y}.png',
                {
  "attribution": null,
  "detectRetina": false,
  "maxZoom": 18,
  "minZoom": 1,
  "noWrap": false,
  "subdomains": "abc"
}
                ).addTo(map_5d87149c73ac45bb9f7cba04c6707063);
        
    

            var marker_97d3f10620f1435dace7083fcef5059c = L.marker(
                [40.69504165649415,-74.17723846435547],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_43f3b90c2b5c41d9ba43e063cf18c4da = L.marker(
                [40.730873107910156,-73.6808090209961],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e29cd9a0b85b481780720391397af77c = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_6d0dbdb88a12400e94a246e8a823aae1 = L.marker(
                [40.6950569152832,-74.1771926879883],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_d69e09e92a6c455d9e4646bbd9eb842d = L.marker(
                [40.96456146240234,-73.80000305175781],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_242591b92bd344eeae23bf7490e1e8d1 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_d19779b4f89b4898bd97f3d99ee979da = L.marker(
                [40.694862365722656,-74.17701721191406],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_bb92ddeb9f7449d19dea02167cfb5b34 = L.marker(
                [40.65434646606445,-73.69132995605469],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_82e9b0f52f57444dbc4b1d69f74578b4 = L.marker(
                [40.72266387939453,-73.69954681396483],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_7adf22cc3ccd4120b632b3a2b355538f = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_5b01c4aecc63495d908f19af816ead3a = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_d50b352980294cd8809c4b7ef0382243 = L.marker(
                [40.53121948242188,-74.39395141601562],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_7ca04a877ad94ad7b964986144d3e72f = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_17790e3bbf9741eb90c346d853c7997b = L.marker(
                [40.69206619262695,-74.18138885498048],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_29386afeddfb434889ad2b6435802fe8 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_00bd1dc67e644c5a920fa2f6f4de27ed = L.marker(
                [40.9445686340332,-73.83106994628906],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_216be6f5bc8543c8b0211fce58109f10 = L.marker(
                [40.690425872802734,-74.17758178710938],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ef2de9338e584dd5a899df94f00814bf = L.marker(
                [40.68768310546875,-74.1816864013672],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_7e87f7bff52447ab9146ac7f69daad9f = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_2895c0ce3fef490a92ce8c414b099416 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a1a81eb7cf404868a280ce11c856bb0d = L.marker(
                [40.69486999511719,-74.17700958251955],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_64bfa97deea74daaa5b15dc14de5949c = L.marker(
                [40.69041061401367,-74.17764282226562],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_3100b7fbc9b44aa19448d09cd4162f8d = L.marker(
                [40.69064331054688,-74.17745971679686],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_2fc552477a4140ef973e6de4472908a0 = L.marker(
                [40.69498062133789,-74.17703247070312],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_7dde63d48e2b471f85fc4e5d1f893b34 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_68b8a936c92c4816bb7b67f59b07a861 = L.marker(
                [40.6877326965332,-74.18166351318358],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_641b8aed1f884335b1442215da3f04c9 = L.marker(
                [40.69488906860352,-74.17704010009766],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_98ffa087e7af431ea2456d7a69ff3d0b = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_526defe83ff54d0b85cd0c8cf1403341 = L.marker(
                [40.57636642456055,-73.957275390625],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_acdb65b855464238af801e5dede4264c = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8d002ba923594865a0ee0693b5749fc7 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_736f7dde832a475ab9aa02f4ac1befe8 = L.marker(
                [40.69480514526367,-74.17698669433594],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_fa91c2845b404b8bbb8e571c5e8ddde1 = L.marker(
                [40.63029098510742,-74.16603851318358],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ebd44b6f4e2f49ccb25e8a31e0185166 = L.marker(
                [40.690128326416016,-74.17794799804686],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_14ea16215ae14e1ca1e2134c6d79569f = L.marker(
                [40.7104606628418,-74.16105651855469],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1b3f6981518f4acd874f1e4f42565afc = L.marker(
                [40.690105438232415,-74.17781066894531],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_12dc9c3be43a44c4bb96ebb731995d77 = L.marker(
                [40.98347473144531,-73.6711654663086],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b2c7fa013e6e412fb0d3f278315da95b = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_d6636d40ae074e5ab3a39325b55eab41 = L.marker(
                [40.687721252441406,-74.18179321289062],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_9aeecabd881443b18f8ae3bc86689bb1 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b61c3e50e1d44488a24f3e672e64b554 = L.marker(
                [40.67465591430664,-74.19883728027342],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_3659a8970bfe45c694991fac1ca0f84b = L.marker(
                [40.70500564575195,-73.65593719482422],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_0028470403434260a85f2c53ff82b444 = L.marker(
                [40.69005966186523,-74.17866516113281],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ec0ad2a42cc7411c9075cfa23a862363 = L.marker(
                [40.690784454345696,-74.17743682861328],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c112b396bfed4ece90af11e76019c7d9 = L.marker(
                [40.81713104248047,-73.68515014648438],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b761ce9c8e9d4895b6b360697dbc1886 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f782938c01074dc3a10d154793d48a9f = L.marker(
                [40.6920051574707,-74.17704772949219],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1b8b876dadb64dc1b69a69928f0b54e9 = L.marker(
                [40.93431091308594,-73.76900482177734],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_6cc18b4061f94a4da7b934aa0f1bbdf3 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_5a6e19cbcf0c4f37bace8d8d03e9a729 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_362cb61b407548659589d9d05ec53248 = L.marker(
                [40.73886489868164,-74.15727996826173],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c6453f16a6724c7ba222b8c7c344e656 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_6a17f4f1cd5c4cf0b77d1a58fd8b6151 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1edb78a719fe4b3f9219f23480b6066f = L.marker(
                [40.687931060791016,-74.18302154541014],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8772b2ccad4a48f69c03a76874c6011b = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e12f36c291b943ada17ee540ae790a41 = L.marker(
                [40.69517135620117,-74.17726135253906],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_7e2234d433b94637b9089d3e329498da = L.marker(
                [41.01687622070313,-73.71812438964845],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_847d14e9e4564ca99f718c739fe68535 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8857281d6de04d05abdd7790198af45d = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a7a871974ae54604b766ee24ad1679f9 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_7310c512326c4429a6d2c2901cbe9ee5 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ab043ebacaa448ea98362f473e363bb1 = L.marker(
                [40.68794250488281,-74.18312072753906],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e5037093b8f349bfaf4172b0bab7a7f6 = L.marker(
                [40.69329452514648,-74.17669677734375],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b3f97492e18d4423856bb05136e34566 = L.marker(
                [40.9550895690918,-73.7357864379883],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_675016cf0e9040d1a4748bb7d525d676 = L.marker(
                [40.6951560974121,-74.1774139404297],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e4e926f78969499ab1740a77500ab649 = L.marker(
                [41.07781982421875,-73.71047210693358],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_d5a3598cafad49ec99ae98e57fce9ee7 = L.marker(
                [40.68776321411133,-74.18151092529298],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f05df7c67b9d4c19a2e9f1acb7b74953 = L.marker(
                [40.69520568847656,-74.17732238769531],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_07b51c6cef0e437d96edc94e72cd835e = L.marker(
                [40.69517135620117,-74.17731475830078],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_28fae3f9bfff4a37a230208fbfde9a13 = L.marker(
                [40.69997787475585,-74.18415069580078],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b075ccf579694750a24dc7da1791143f = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e4e3ef453a374d58b905170551cab189 = L.marker(
                [40.694923400878906,-74.17715454101562],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_292632c4530a481794abb63e4f32bdd4 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_4ebd6be5efa144f4a1bc9567c2b8fabc = L.marker(
                [40.55345153808594,-74.302490234375],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c321f8c6f31e4611853d7ae76fd1f85d = L.marker(
                [40.69561004638672,-74.17827606201173],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_4aeda05f702c493aa2f845d8ee7690c2 = L.marker(
                [40.934608459472656,-74.09500885009766],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_4ac8a7632efd417ca274aba0f07c1a21 = L.marker(
                [40.69538879394531,-74.17758178710938],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_9c987e5a936445bb8b7e04f066f51450 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1284e8740b2442bdad00e5cdaeb4bd63 = L.marker(
                [40.60683822631836,-74.1644287109375],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1065a5cedcf6472fb8ad733d53f86eb6 = L.marker(
                [40.56480026245117,-74.18228912353516],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_6fd331acd95c4708ad7dfe7a75f4276c = L.marker(
                [40.934364318847656,-73.84799194335938],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_23c6121113874801a16fa06bf1dea8a0 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ff40655138cd47b7994225294ffd5000 = L.marker(
                [40.935455322265625,-73.90299987792969],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_4033cd5c5a64459aa247b6e801e75f31 = L.marker(
                [40.68924331665039,-74.17867279052734],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8c32b0dcfb89413e9a44c83930b20691 = L.marker(
                [40.69445037841797,-74.17681884765625],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_cd1a364abb5541c1803cee41b315e2f4 = L.marker(
                [40.68769836425781,-74.18204498291014],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b95b64fe6ec54e0aa8866faffddf371b = L.marker(
                [40.691707611083984,-74.17707061767578],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c4c54e7a3f0340a8ac0e9bdac6781c89 = L.marker(
                [40.694068908691406,-74.17683410644531],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_34e6adf78cce47ebbaef7cd6ad14d603 = L.marker(
                [40.69314193725585,-74.1863021850586],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_636fe0c701de4349a35792c6f5ebaa8c = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ee5034e7c57f45699dfc658f0c2169d5 = L.marker(
                [40.695533752441406,-74.1778564453125],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_42ef091c31824e1f908023dbc14bf535 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e8667ae7d0d64cf2be2bde569892d0bb = L.marker(
                [40.69550704956055,-74.17784881591797],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_4b1dac7bcf204d9384bd6f15586cc5e1 = L.marker(
                [40.57393264770508,-73.99617767333984],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_d3e20e3f6e06499fb5f7d5fe90ae0f60 = L.marker(
                [40.68825912475585,-74.18350219726562],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c80ad9c620ee4ceaae22e6c0e2c1d03f = L.marker(
                [40.68795394897461,-74.18297576904298],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ea6c2b7bdcc84d81837191bd7df44549 = L.marker(
                [40.71900177001953,-74.35848236083984],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_0deab263281c4fd2a8e1b8faeaa07350 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_80d5ce9831714a8e8526e1a3ba06eeb6 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_57bcaf2cbf1044caaec41b2fa19d136f = L.marker(
                [41.03084182739258,-73.59837341308595],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f1ceb55469c34509b653830cab722ea5 = L.marker(
                [40.74651336669922,-74.1646499633789],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b17bd5b4878a4e95b18834014aeff361 = L.marker(
                [40.69032287597656,-74.1777801513672],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f3598525b1e645b4b6e8372351ed3c6a = L.marker(
                [40.694969177246094,-74.17708587646483],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_eb86171b4159423a92a344093a9f7e9d = L.marker(
                [40.69548797607422,-74.1778793334961],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_0791e18cdd7849979e13941dc2455350 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_0fb1948ab1c140b3b0ebcab82b3ada22 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_50dc313186b4434f861937ccb4fb14b2 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_3bb30306149048a6b8f0bd4982d18d7f = L.marker(
                [40.614784240722656,-74.17701721191406],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e93a3db2722a4c45b4cb6a05c68dd5e7 = L.marker(
                [40.68769836425781,-74.18180847167969],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_af760f28811d4662be7ef2cce712e44c = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c0f140ffe6244034bfdc5cc88a8b76f3 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_2f34f976b02b4178bbf37fe1cf766714 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e594a6b849b64331a48ebfabe9ee0ceb = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_932a1d0414ac4fb593fcae34e1f7ca8e = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_cb1f91436e0d4bfc8388373f7cc70d9d = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_cca8ac441488429faa3dffeb0b2ff24f = L.marker(
                [40.69488525390625,-74.17711639404298],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c8733a9ed25b4c778ec9e067906ca481 = L.marker(
                [40.694820404052734,-74.17716979980467],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_00ae9a0f5c79445c94995503794699b4 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_48d94b4f15274575a483eb5a2d49f873 = L.marker(
                [40.68804931640625,-74.18324279785155],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_605116cae6af4fc9a47e51fd303bf43e = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_19b92f117a064f1eb3acf7167be26ed4 = L.marker(
                [40.695655822753906,-74.17841339111328],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_9ac2fcd643bc443ebc05d3098c587501 = L.marker(
                [40.50335311889648,-74.23754119873048],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_bd1198ba8ba149479655f26ffe5d40ce = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_d6b91b3aede0499583858ad3bcac7857 = L.marker(
                [40.69498062133789,-74.17732238769531],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_838f95b74f41448aa15ec51a1cf114b9 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_9390fa130ab2445b8b0802e3325bd056 = L.marker(
                [40.76486206054688,-73.53269958496094],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ad7703a1ffa643ea886eebd8cba2161e = L.marker(
                [40.69529342651367,-74.1774139404297],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_061c26419a2043c0b95004d87128fff8 = L.marker(
                [40.68775939941406,-74.1814727783203],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c8453d0bf5c24290b71640403e5e3f18 = L.marker(
                [40.688323974609375,-74.18361663818358],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_2e5099616c354f2da2ff4ab753d20537 = L.marker(
                [41.16884994506836,-73.62869262695312],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_671eea492dea428283166e4727f26be1 = L.marker(
                [40.68854141235352,-74.1838150024414],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_3e8867fe3674443690228a87503c4bf1 = L.marker(
                [40.69314193725585,-74.1863021850586],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_3f4c74bf7c804f17b96f4cbcedd8a2b6 = L.marker(
                [40.572132110595696,-74.1210708618164],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_6bddc0c2da8e4cbc9dfcf33fd8007cb3 = L.marker(
                [40.752601623535156,-73.55078887939453],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_2173fe99962845ca9e124fc94d16c533 = L.marker(
                [40.693641662597656,-74.17664337158203],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_641b92a39f9e4e0d80024ff6dadecf05 = L.marker(
                [40.57542419433594,-74.16191864013672],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_869131c7944b4509a3b01b47f13181b1 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_793f876fe6f1400f95ed0733dd4a2867 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ca8ab9db132f4c679613cac69dc58141 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_6f487aa0556a41629e9cc36045310c96 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b919f718417a4f12be1882da7ff6f2bb = L.marker(
                [40.632240295410156,-74.1521987915039],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ac087733ddb74c96b886d5cbbd215fe6 = L.marker(
                [40.70984268188477,-74.17779541015625],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_dcce4b47955a4ed69e5aa64c763107b6 = L.marker(
                [41.03572463989258,-73.8582992553711],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_325272a30b784fada0c18f13da4bbfa2 = L.marker(
                [40.69543075561523,-74.17757415771484],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_963b79f5b07443ff9721de11073dd140 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_15f51f5ae10d4146b57413b5ab49ce00 = L.marker(
                [40.69026565551758,-74.17777252197266],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_0273f3e3b4c64255916d3fec66706c19 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e87eb27db17a426295ad525becc3c8c8 = L.marker(
                [40.699703216552734,-74.18611907958984],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_84bedc59b49e47beb87226ac52f3c526 = L.marker(
                [40.69314193725585,-74.1863021850586],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_11154544d7324378b53e268bb83f56f2 = L.marker(
                [40.587100982666016,-74.15374755859375],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_922ddaf4fff0419b87133863d667e249 = L.marker(
                [41.09019088745117,-73.91571807861328],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_fcfecaa6deb7482ea327c8656daa5051 = L.marker(
                [40.69992065429688,-73.54464721679686],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_413c87de42c343f29eb2f940c8cd91f8 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_588714afd4634c28abbcbd6b0c5f5a9c = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_4beff1b710c94a1b8a5b85620498f1cb = L.marker(
                [40.68774032592773,-74.18167114257811],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c6ce4b689cbe4295804e0c8a3f94e9e1 = L.marker(
                [40.8162956237793,-73.46357727050781],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_99dfe233d0c74603a5ba869b18971003 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_fd0c291ac50c46148484b76c1f30d566 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_2ceda2d9e0d642c18e9ac669550c5a25 = L.marker(
                [40.978370666503906,-73.78406524658203],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_6ff3c822bcf84982aec0a9b524fe89f7 = L.marker(
                [40.69517135620117,-74.17723846435547],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1956ae2461e84f0ba04b6452356d55d3 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f718dcadf1fa4367adeffb1568fe100f = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_3c2b73943c7841d68346b848dd2038e0 = L.marker(
                [40.695411682128906,-74.17826843261719],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c3a5336db13c4c6a817b30233c363764 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_09ee4d04b97e462481d16354ef5c44b7 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a94b5965b146491487fca7d9bc3e8ecd = L.marker(
                [40.687767028808594,-74.1822967529297],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_0b26c8b17c96456fa90c35dbc7fb015e = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ac512d810daf4a7f9f3526aa012fdee1 = L.marker(
                [40.69088745117188,-74.17729187011719],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_888aba03f1e841ddbd57d1f380d5e8ef = L.marker(
                [40.55461883544922,-74.19615936279298],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_56b09a5af9d74f87b91ebc77276c48d8 = L.marker(
                [40.71229553222656,-73.68190002441406],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_4a473060b6674edfa6268f6dd5682f39 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_575149622ba149eebe61abc1ec4bda47 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8b3d79b3bcab4a6583d145d2f63ca8bc = L.marker(
                [40.74885177612305,-73.57504272460938],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_bbe0c2e5869546eca894f62ae9c505f7 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_59c06375919642c9b2f74634c7c68d12 = L.marker(
                [40.69066619873047,-74.17755126953125],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_efef3dcb31d14b26bbf9287baddcf57b = L.marker(
                [40.58675003051758,-74.16533660888672],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_2da74fff9c5740d2a8347ffe2df7264e = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_19deacbcf8d24de2ad32a020035cac90 = L.marker(
                [40.68923950195313,-74.1788330078125],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_71a15b0fccaf4e59ba30478396b7e56c = L.marker(
                [40.69517135620117,-74.17742919921875],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8a49fe29b6d946768a906106ccdf9dfc = L.marker(
                [40.69543075561523,-74.17759704589844],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e4e6affdc8d940a8b9fdeaa68e6d2c02 = L.marker(
                [40.69535446166992,-74.17742919921875],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_443567c1629d4ef9a237073f4804122e = L.marker(
                [40.69499588012695,-74.17697143554686],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f944eb1f011f40c792752a4cd52ddad3 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_9ef9fe04ed804a26a93a22ef1582d216 = L.marker(
                [40.68771362304688,-74.18190002441406],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_0bf5c806dff6457da47e54f1ecc7896d = L.marker(
                [40.690914154052734,-74.17730712890625],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_04111773e05d4afea0a4d2a653552aa5 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_d84d26b5d0fe4c0998f748cde45df8f0 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c79b982402164f4e8931de51d643b2b2 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_851bfab7d069493e82984d476625318d = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_7639922d924844fb9c6123ae5b2c2023 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b26b2d8ca8c6415d86a804dc5ee63054 = L.marker(
                [40.57457733154297,-73.86189270019531],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f878278c1dd343938cc530266a827aed = L.marker(
                [40.69543075561523,-74.17784118652342],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_0b36bccec34f40339ac28c7e389c406e = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_9e082c49a3704929972484d8221436e8 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_4ae197a04cbb40149ca620a3d9c52a9f = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_201b5f29fb5847178701ed24ee00e515 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_dda4b4b20003464e9d233f0b86374d02 = L.marker(
                [40.69549179077149,-74.17768859863281],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_3805968e582a4dfa88653013407fef7d = L.marker(
                [40.6346549987793,-73.69601440429686],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_dc59f01b949e44c294be1bb4238a1eea = L.marker(
                [40.7130012512207,-73.68321990966798],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_3e2a1e2caf7e41ef96fe6f339f0f42ad = L.marker(
                [40.69522476196289,-74.17721557617188],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8deb9114dbde4e0c9b653f992a69b071 = L.marker(
                [40.68785095214844,-74.18284606933594],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ab794cbdb4984a4086b1404e003e756e = L.marker(
                [40.68770217895508,-74.18167877197266],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_5d9e0fc7f7c549c0acff982918ee8ec3 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f8ed69dfb2754d9580470d50eac7c32b = L.marker(
                [40.57563781738281,-73.95596313476562],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_74fc736349f749d7bb963e2bca8a21c8 = L.marker(
                [40.688087463378906,-74.18335723876955],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f386faec361c4aacbf8324dcd72c7099 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_2fd856a75481467d878bdfb51ebf0591 = L.marker(
                [40.9783592224121,-73.77363586425781],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_9dff5b41b6174cbfb5ab9396ba44f410 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8b942dc8479847fc9084f2532caf194b = L.marker(
                [40.57591247558594,-73.95501708984375],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_65c852cc46354296a0b23382e0be7f43 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_cf74eb1aac6245489daf89bb097180d2 = L.marker(
                [40.662841796875,-73.67752075195312],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_5652150b1f2f4013a3d0ba362bc0a5a1 = L.marker(
                [40.69038772583008,-74.17765045166014],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_dd6080d8487f4cccbc3e31b530f8b482 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_362de2fd41894e4490633fb64b3a5f82 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1fba8956ea204b4f8c4787e42b7fb06d = L.marker(
                [40.69446182250977,-74.17676544189453],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_6402b7296c994399a695264f2e6b5146 = L.marker(
                [40.57719039916992,-73.95478820800781],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_36b6efc0c947421a8bd1fec968734100 = L.marker(
                [40.80049514770508,-74.2894515991211],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c3e4544b149c49058ac3fd52f7211ad1 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_016c9628fb4a4840ac05fb13ffd7f463 = L.marker(
                [40.69519424438477,-74.17727661132812],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_bc0e75474ac54a619479f7cd5284599e = L.marker(
                [40.57524108886719,-73.85546112060547],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_0196a13f45c242f5a55d5780fec58154 = L.marker(
                [40.57471084594727,-74.00798034667969],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_af13ec23a3974b1b8b5ff6e1106f94f5 = L.marker(
                [40.69145584106445,-74.17704010009766],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_869987322bee454aaf9689b31927dd0d = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8eabdd89807948fab51fd552a3e55832 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a2c13e75788f43879431caee49584864 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_786e091b7e3348dcb17a2c44c722bdf5 = L.marker(
                [40.68767166137695,-74.1816864013672],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_057eb153ecd2475bbbf2ca9a894ba19e = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_da1e432718fe4b7198340210b254408d = L.marker(
                [40.6907615661621,-74.17739868164062],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_9cfc75a827d54434bd481dcc1710d3ad = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8a1caa594909464b9b777e9986647de0 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_de17c6ac03c1482ba06e7a9f14402bc8 = L.marker(
                [40.68792343139648,-74.18309020996094],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_989107ee6cca49828041051212d6c5d7 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_35c38fce13ea4cd4a1ade00b2bc84a0c = L.marker(
                [40.93616104125977,-73.84590148925781],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_6f24192b25224208bf0ae0c9e156aa52 = L.marker(
                [40.69501495361328,-74.1771011352539],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_09ac0ab6825843f5a122a1a654d72410 = L.marker(
                [40.68791198730469,-74.18291473388672],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_74d919a7e1054eafa8c7f6aa5483832a = L.marker(
                [40.68765258789063,-74.18208312988281],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_94d128935c994eb284ed5a3f6f113265 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_4a5ac9676f3943969a47824c0c57f4fa = L.marker(
                [40.69083023071289,-74.17731475830078],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_4a4e2df7c71c4df7ba8da7cdfb584f71 = L.marker(
                [40.68888854980469,-74.17915344238281],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_d8190a6087fa46bdac9d0e468b124b86 = L.marker(
                [40.92839813232422,-73.75747680664062],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ce4fb6917cbd4b0794c4129ff2ab10cd = L.marker(
                [40.6914291381836,-74.1840286254883],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a8bbf1bb03424a6189eb0a42f193ff2e = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f2db0e1168bf49c8bf8ccf187588236b = L.marker(
                [40.687965393066406,-74.18313598632812],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a5d0c95c72214c1795230d47aaaf8432 = L.marker(
                [40.7418327331543,-73.60811614990234],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_d50ee7c1cec94634bb86bec224efb14e = L.marker(
                [40.948184967041016,-73.89813995361328],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f43982fde68146898dc31489b0f3a989 = L.marker(
                [40.6943244934082,-74.17678833007811],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_426a505351a24fd3849f9287801ac149 = L.marker(
                [40.69051742553711,-74.17757415771484],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_7164273abea1444093375e767fc81205 = L.marker(
                [40.81462097167969,-73.47482299804686],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_02192c18ca914002a3f052c19c106b94 = L.marker(
                [40.70806121826172,-74.15289306640625],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f7dc18972a9e4a538a6754fbcfef368c = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_557e274630ff4df88c87870a60744af1 = L.marker(
                [41.0793571472168,-73.68404388427734],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_76216eee890d4e718d02fd49de430b4c = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8d6c690930484d8ca44235ad702cdfe1 = L.marker(
                [40.68858337402344,-74.17961883544923],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_650c073b12404a7880b47c2e8f71b803 = L.marker(
                [40.6880989074707,-74.18123626708984],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1b2e8066f30148c287e302b9e2911c1f = L.marker(
                [40.69531631469727,-74.17739868164062],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_2e37f4d4350042e8940d1b6fa69dd446 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c4580b36561e499e871a857f70d97c7d = L.marker(
                [40.69519805908203,-74.17733764648438],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_9553208c90a84d6186ea0fdc82207ce6 = L.marker(
                [40.51942443847656,-74.19486236572266],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8d26f3a0488e442e8a0052b3a106e96a = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f808e22bc0bb4e51847cf7ab35d4a63c = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8bb167d1c6604863b347708667cb3ad8 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_d5657d84539f4718ad4ed88bf53b548b = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_00390812dc6440fab7300949c1b13e81 = L.marker(
                [40.69498062133789,-74.17697143554686],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_51f3bb69573e4d57b40517d604b6da5a = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_bd1ce1227efd4bdea5b76df4382ec9a1 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_d9ebb6f730084fd7a0a5fc4655bda19a = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_5773933f01844883a7d12e294fb776bd = L.marker(
                [40.69495010375977,-74.17709350585938],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c1e395934ea7498e9439e46fd45e839c = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1fb5bf305dbf4949be84239dee84c127 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_34de2834701b4c94bd2881e2b7a51da5 = L.marker(
                [40.95464324951172,-73.72296905517578],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_0c22a73da94b458f87228de7b0f80ce3 = L.marker(
                [40.69060134887695,-74.17752838134766],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e993690eda604e979c3091e28f593d5d = L.marker(
                [41.084617614746094,-73.78501892089844],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_52e672886f304bbb8d15f473d758b143 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a40cbca920854a02a5a28c0681311281 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_6407cbbabc4a48ac9e747f77158acfdd = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_6a1e4d7b8bdd402fb73d6eccf96ca5c0 = L.marker(
                [40.69508743286133,-74.17718505859375],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e6e0f7846cc744e6bba37f95618b514d = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ea460190ebac4970bde2730ba1d1b9c2 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8518da4c949f4b679015a533af5301e2 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_2e9b0feb85ad43869d6ca04d23c82ba0 = L.marker(
                [40.86786651611328,-73.4326171875],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ba33e146f23541be8410dae256d72683 = L.marker(
                [40.69488525390625,-74.17724609375],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_277075d1199349309d093731a006876b = L.marker(
                [40.69969940185547,-74.18387603759766],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_86914139dcad47f488b77afc6308902e = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c1cc4395c26d4a9991d557891b5bc6e0 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1eb129d5af464ec28e7e791dfc29a6a8 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f08b95a8bc224e0198a6e73994035b6a = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ff54ea845eab4e0b9a440070f81d76a6 = L.marker(
                [40.92622375488281,-73.75233459472656],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_45d2fc9b0ded4111bcff613f912c4fb1 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_41bba5815c50431b90771ac2a78a5ec2 = L.marker(
                [40.695228576660156,-74.17740631103516],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f2f299a64520424eaf2ee3746f18e15c = L.marker(
                [40.69486618041992,-74.17704772949219],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_6a502199cb70474ea4817b715ead40d2 = L.marker(
                [40.6900634765625,-74.17862701416014],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1eadb370450a4ba79f0a2ca0fe46d742 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_0e86fbfdde9d4d1dbe92c42f6aa3f16d = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_6fe7f8321853468f94bb6b7fb4828813 = L.marker(
                [40.587013244628906,-73.67828369140625],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ed132688007c48c08edacd08131b9e30 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_aac5338742554147ad0856dcb702a47e = L.marker(
                [40.6943244934082,-74.17678833007811],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1bb57ca50bcc435f9a79c910a02e7d99 = L.marker(
                [41.22004699707031,-73.7237777709961],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f4f3b3eb01bf41ee817a17a7ea57e3b3 = L.marker(
                [40.687686920166016,-74.18230438232422],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_cdb66e4bf5c540e092e29dc1484274ad = L.marker(
                [40.695499420166016,-74.1781005859375],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_5f0ceaa82cc745c99702a7553faa28fe = L.marker(
                [40.71490859985352,-74.36174011230467],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_664b4ab7ded44d9085c986da46b79c33 = L.marker(
                [40.69517135620117,-74.17723846435547],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_51a6049fd73a4a43b2c2be58f866c941 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_4bc57cd228e849c29f8797c6fe6505e7 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_497d7992f6b040beabf31a9f471571a5 = L.marker(
                [40.69511795043945,-74.1771240234375],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a2a36cf45b05416e92a68b4bb46fd1e2 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_6aefbd3d99864ee3a0608ffb7e1415f9 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e866b7478e044eafbc5cc7aab7a0108e = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_3690a9510c334385b85150dd40b05f66 = L.marker(
                [40.69098663330078,-74.17750549316406],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_82acb866cb904cd782957aaec2dde08b = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_03cac051313445b299b3a129e7d13e38 = L.marker(
                [40.65571594238281,-73.69139862060547],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1283a7a5cdef4899b9b2cf17bab15e6f = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_dc460a6410904fd39cae5d7185a7324a = L.marker(
                [40.6945686340332,-74.17681884765625],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_bdb3881f4fa94c7f95ab9b7781149b92 = L.marker(
                [40.71115112304688,-74.1776123046875],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8d91b114f9e24b2c9c829b5c0827c87e = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c8760f099b584b669125f5b67e5d4e9e = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_2190d5023f1e415da16739722b0e936b = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_bfd855bc11a04065831fc8041df22921 = L.marker(
                [40.687721252441406,-74.18155670166014],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_20ac27d280b74abdb02c7a1938ac94ad = L.marker(
                [40.95466232299805,-73.8628921508789],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f0bb1fbf50744b7bbc2267125edcab4d = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_fa37fb3504f94de7bbe1899bd0dbe431 = L.marker(
                [40.69453811645508,-74.17681121826173],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a80cd475a8ce46d2b24e96acc8481e1e = L.marker(
                [40.69514846801758,-74.17723846435547],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_19378b5b947a4480b3840e7f1e124ca8 = L.marker(
                [40.695472717285156,-74.17774200439453],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f436bb08c7764d8191e601f4972e3c63 = L.marker(
                [40.69057846069336,-74.1775131225586],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e742fcede91f44a1a003dbc13b407aff = L.marker(
                [40.694801330566406,-74.1771240234375],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8a8dacd47c7e4dcaa93b31daeac48760 = L.marker(
                [40.694786071777344,-74.17716217041014],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_bb924e011c404e62aff10890ce5603e6 = L.marker(
                [40.693843841552734,-74.1766586303711],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8e8964bb0eb2412f8efc360329f63e10 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_d1adaefb17fb48cf831d959d669d0f43 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_21d681b95bea48edbb1bdeebfae64383 = L.marker(
                [40.69314193725585,-74.1863021850586],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_50e5aaecbca54e75a80afcd17228bb7b = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_467f01f6dc8e4fe7b48ee6afc16f187c = L.marker(
                [40.69477081298828,-74.17695617675781],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b7044386b55d4affbdf9c986cc28e901 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_46994ab6ed2b4207a50f20cf54b0296c = L.marker(
                [40.68650436401367,-74.19308471679686],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_bf883f89a3a84980afb4c69f5d1965a8 = L.marker(
                [41.03560256958008,-73.7687759399414],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_525f0e6a645142f1843977c12bb9bcdd = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_d0ebede835c641fb9f44c39a4a6400b6 = L.marker(
                [40.695159912109375,-74.17755889892578],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_67cab0052c794ad7b98977c9ac3c9515 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_26887eb0bc814bac9c1441fd9fc78a69 = L.marker(
                [40.98584365844727,-73.81060028076173],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_208f80fefbf14cb7968f821dab569136 = L.marker(
                [40.60417175292969,-74.22815704345702],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f641364abb384bb686a85cffecd0f442 = L.marker(
                [40.695411682128906,-74.17757415771484],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_143c285dcd134cd19e693a520068de37 = L.marker(
                [40.94255828857422,-73.8361587524414],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f44c03807664432caa251b5664e69d84 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_7f9fd52eae60464584b061aead0201a4 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_439ad2efdc47481982e4437343eb41ad = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_91cf0d7d31ee4311a4d0940825186aaa = L.marker(
                [40.68777084350585,-74.18126678466798],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_d304467d0cc84ef69d26cff4f64d92ab = L.marker(
                [40.6904182434082,-74.17768096923827],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a165bbb33d464cc49594fc8c66c399c6 = L.marker(
                [40.68238830566406,-73.38333892822266],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b66b7afc2f0b4d3eac83ae8722bcea59 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_dff39bd4c79b459e94363d556c04c9d2 = L.marker(
                [40.695457458496094,-74.17771911621094],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a739a094113a4fea961df86663bce538 = L.marker(
                [40.69493865966797,-74.1770248413086],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8dccd36453334b7aa09006f69105023a = L.marker(
                [40.68810272216797,-74.1834716796875],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f1b50c7002194ca6b13ad42c3f631a53 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_efe0cbe635474e9fb6f1951b0f07c470 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ec9a1c83e26a495f9ed1875c75b5439d = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_d7532c7fd1e241488c8aa840622b858b = L.marker(
                [40.69053649902344,-74.17755889892578],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_583e7a1d23fd4e9c9c9b055765f41f94 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_3b1f414af08d4265b80240f91180ece2 = L.marker(
                [40.69519805908203,-74.17728424072266],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_4303158264bd47c3aa6f4bdb9ea2bac3 = L.marker(
                [40.77726364135742,-74.33901977539062],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_afe097d75af9477c82db22d58393238f = L.marker(
                [40.68771362304688,-74.1825180053711],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1859e2267a944e19965a017b963bab71 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_7f657887e27741eea90bf9eaeeb27b86 = L.marker(
                [41.01926040649415,-73.72421264648438],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ecf3c505088c4772adb58b68fa49c227 = L.marker(
                [40.68980026245117,-74.18425750732422],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_24dc07ba97c14e808d0a7e49f7360957 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_3e63e53816d648cf9a7844231119a9b9 = L.marker(
                [40.745941162109375,-73.67893981933594],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_37f76156e70c46bc87f96a3f90469278 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_99d0ae9735304e1da22f9182a03342e7 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_3d191ab318f24c73ae274159da042dba = L.marker(
                [40.58549499511719,-73.67076110839844],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_51c9d7c6ce4846739223319e018833e3 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c2e3c7d9a8bf4ef0b2c2c534ec5f62f3 = L.marker(
                [40.68770980834961,-74.18148803710938],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b27a5f00721e4161a333e30b5b899d4c = L.marker(
                [40.69543075561523,-74.17765808105469],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_86c27fcdf29347e1bd7be29ca2d652a4 = L.marker(
                [40.688453674316406,-74.18347930908203],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ad90a15840fa4b7eb3964a762572ee7a = L.marker(
                [40.6947135925293,-74.17768859863281],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_211143c48736451888cfcbda5efd2fb8 = L.marker(
                [41.03281021118164,-73.83438873291014],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1ece5df6e3b74add90a8709a2ebaa6b0 = L.marker(
                [40.690444946289055,-74.17755889892578],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_9e297b564195495a9a813a74e717c194 = L.marker(
                [41.001949310302734,-73.75090026855469],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_58109de0919f4082a4ef2c371eeb6e1f = L.marker(
                [41.001949310302734,-73.75090026855469],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_14f8b1b114744957a054cf6ff0d5e4d4 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_de00a97ee1d14119923917ae1db10ef8 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ec5754d489204fddac421229041ae584 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_81f942a156434c2db05087664beafefe = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_43a1dbc9833d409ba3c517aa9c42125c = L.marker(
                [40.9420051574707,-73.83922576904298],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_7dc5e6dda5dc4e5ca2eaaf6fe467f550 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_95cea043381a4ae2814e5ac5d738ec53 = L.marker(
                [40.69146347045898,-74.17707061767578],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_0f43ad3113774ef4bd84ecda6835ad93 = L.marker(
                [40.695274353027344,-74.1773910522461],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_14ff1026622048d0bb86c13dccd879b3 = L.marker(
                [40.69110107421875,-74.1771926879883],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_424e3cb620424e5cbb1d719a84b5ee5d = L.marker(
                [40.76832580566406,-73.52824401855469],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c025eabea6d84acaad6e398d77be14c9 = L.marker(
                [40.69314193725585,-74.1863021850586],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_9dfc9328d40c4b09b40372523d7adb4c = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_513b1a3eedfc474bab25e4ff5f29f671 = L.marker(
                [40.70729064941406,-73.55044555664062],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_977a724338934e31a842305491390629 = L.marker(
                [40.5701560974121,-74.11410522460938],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_afc76d92c65943bdb016ed08b6984e51 = L.marker(
                [40.69018173217773,-74.17794799804686],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_08d0acd1feb845c39798c36776413057 = L.marker(
                [40.71033477783203,-74.16078186035155],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_94fe6d43e2064a29b3dc41617231da8f = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_0a97c8955a464ba6a797b624d5ba889e = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_555a1ebbfb2349cc90333bc4bbc292d0 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_7f150b8f9fef4e3785fd2d4bbb8a988e = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_17fd11aa8f9547b398d07792bf98619b = L.marker(
                [40.70105743408203,-73.65438079833984],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8483248b2d8c4cd2817d0e6818a5f596 = L.marker(
                [40.57278442382813,-74.09383392333984],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_14d7fb74364a4201977d9dcad40c0b49 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_007c4c6f7d4d4b3eb770298e44b5df6e = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b2a6c8dbc10f4aaf98600b419b426951 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_41188211d5d7475cbbeb0a8951231996 = L.marker(
                [40.693225860595696,-74.18671417236328],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8d0f7475e49946d0b7c5dbe07a9fc2a7 = L.marker(
                [41.06093978881836,-73.86284637451173],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8705a07837984c2689e54d5a43cfbde5 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_7b9c8fc3773a4a44a0746b695d329df5 = L.marker(
                [40.69387817382813,-74.17668151855469],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_57e38d9ab0ba480e9cb115f876120fbe = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_27e05ef3efed4f3191572b0c5a864a30 = L.marker(
                [40.68770217895508,-74.18231201171875],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a481b596cfe7485089963925e2150829 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_097076b129cf4d51b7b6d097e6cf6724 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_97e25574c6924b2aa1fd2629b8f4fc46 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c7942780c0aa4251a441424dfd11c0eb = L.marker(
                [40.75527572631836,-74.25259399414062],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8423afcbcdb64265a1b38fdc87aa563b = L.marker(
                [40.80524063110352,-74.20175170898438],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_bcaf8d9d9717497696685f02548fb8fb = L.marker(
                [40.695125579833984,-74.1771926879883],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_005426704f3240b1930134f5099f7240 = L.marker(
                [40.6904296875,-74.1775894165039],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_41bf6c2aa04140d7ab65eece8219165d = L.marker(
                [40.94001388549805,-73.80106353759766],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_429244ec5de4445380cde178f0f7146d = L.marker(
                [40.6951904296875,-74.17736053466798],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8bcb2459d9dc42f3801e69eae8590471 = L.marker(
                [40.57372283935547,-73.85069274902342],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_4e7dcf9d43c34badacd266fe0c56e50f = L.marker(
                [40.69511032104492,-74.17728424072266],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1f58a338cd11432eb083380051913ee0 = L.marker(
                [40.6901741027832,-74.17772674560547],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_642dc95fc3254e0191fb39d4f39eab40 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_4652403447a44456b32b545f69b6cb88 = L.marker(
                [40.69049453735352,-74.17755126953125],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_0108efe73fd94f688556d5e2377366c8 = L.marker(
                [40.69521713256836,-74.17759704589844],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e7062405742e4732bec2442b570d6c6d = L.marker(
                [40.749183654785156,-73.6687774658203],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_eae8141686944c6bab365d76173eceb3 = L.marker(
                [40.69021224975585,-74.17774200439453],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_0b8a732a303d4e8099f226a5f7690244 = L.marker(
                [40.69005584716797,-74.1778335571289],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_731994227bfc44da902de45cf5abff51 = L.marker(
                [40.971015930175774,-74.14189910888672],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_002c3833945440e3b071fa6aa83be8cd = L.marker(
                [40.71370697021485,-73.5988998413086],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_01aede22c3f640f49c035f10138c5bb0 = L.marker(
                [40.690330505371094,-74.17765808105469],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b6afdc1b342e4512b4de15b5a5df2e51 = L.marker(
                [40.6905403137207,-74.17755889892578],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_2976ea2958bb48fdb99ed80d82922237 = L.marker(
                [41.05854034423828,-73.42540740966798],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c2285b4d685d44db93961cf3d30729f6 = L.marker(
                [40.68769836425781,-74.18145751953125],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_6ebf15e3562049c4823ef53b83aede29 = L.marker(
                [40.69336318969727,-74.17678070068358],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1c6f9d212a1e4b14b1e85b4ac3db6edd = L.marker(
                [40.93547439575195,-73.8188247680664],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_037ca84a01cc4aeb9d35e7f59238acdb = L.marker(
                [40.73394775390625,-74.24674224853516],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_3bf9dde903e8454eafa76510238b6037 = L.marker(
                [40.6910285949707,-74.17723846435547],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_d771cc9ae3a84e69827290c207881afc = L.marker(
                [40.687767028808594,-74.1815185546875],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c1c9e385b416400ca1273b873367832f = L.marker(
                [40.68834686279297,-74.18020629882812],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_5401a866b8714bfa8731e19b7e8fce5a = L.marker(
                [40.78632354736328,-73.6746597290039],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_bbacc54450914fcca9405f0c129abca4 = L.marker(
                [40.69153213500977,-73.69986724853516],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_7617193051574ac682fdf99a88068fa4 = L.marker(
                [41.026283264160156,-73.6240005493164],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_d880dba54fc44c16ae9a707861e459f1 = L.marker(
                [40.690818786621094,-74.17737579345702],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_435348fbcab24ce09e4e239c4c8497b5 = L.marker(
                [40.75210189819336,-73.67896270751955],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_97e9a86c64294dacbc82608348b414f5 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_623f46d581eb451aa945f63a63571778 = L.marker(
                [40.7116813659668,-74.16447448730467],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_86492e94719f4812aca560a0e0bb80b7 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_421add4c82fe42179d3d9c8f4aad509d = L.marker(
                [40.77673721313477,-73.58558654785155],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_38d0ba2642f243d786b2020cf9148be0 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f49b08db470b40e2ba2ff9e9eaf2f67e = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_0a3aed3bd7304f4e8b2d8a8733452642 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_2bcec3e2c3bb4a088ca554dc2f96cea8 = L.marker(
                [40.69211196899415,-73.62572479248048],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_17766d5b0b754824befb2128faf0bbc8 = L.marker(
                [40.75065994262695,-73.60391235351562],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_480d78b5358947c290c56bb8beaca4fe = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ebb60ac1223245a7a7a2c4fd295a36e7 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a58708c036d0439e800449214cfb1945 = L.marker(
                [40.91970443725585,-73.84686279296875],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_fbb29d07e7a64a66b8b5c4cabe41f458 = L.marker(
                [40.69314193725585,-74.1863021850586],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f9fef9966fc44e79b8dab16ef5c78ddd = L.marker(
                [40.69279861450195,-74.17676544189453],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_dc54863947244a8b99c73d9d3cbcebe7 = L.marker(
                [40.7032585144043,-73.68514251708984],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_acc73263a4314390a04d067a166cb4ac = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_9ed31ca1efa44229a8751932c6eaa8f3 = L.marker(
                [40.692970275878906,-74.1765899658203],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_5c2c8880d36c46ad84dea25effc1e463 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_401fc5e127bc42eb9e7ca9421a38d397 = L.marker(
                [40.65010070800781,-73.5646743774414],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_231df991d05c4c04a94e0ecc1db61e63 = L.marker(
                [40.69060134887695,-74.17750549316406],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_cf12cffc025c4f31b258bac7d2121905 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_3363195a35084abe922d67354169c6c0 = L.marker(
                [41.053260803222656,-73.5465316772461],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b7f824233ae447e789b36508ae376d04 = L.marker(
                [40.69545364379883,-74.17779541015625],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8a7caf0bd7774249be89b3792c44f6f1 = L.marker(
                [40.615631103515625,-73.62976837158203],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_90080349b305444f8e9004b05a8c92d1 = L.marker(
                [40.69113540649415,-74.17716979980467],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_3a440e938ebf474a8c51690d8b9dea97 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_496f5e931e3643408e63d2368d6ef814 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_065bc2fa2e7e4960b588acc98f610fe6 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_64963115d227419ab13b173f4fd53431 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e2e0b2caebe34f19b89359db631a53c3 = L.marker(
                [40.852874755859375,-73.41123962402342],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_aa86313689764cd68df617e4684b47bd = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_9715a1520f224449942085078a753d9d = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_0a8544d62a1e4d6a8404e92d33bd5481 = L.marker(
                [40.91964340209961,-73.83025360107422],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1984728e259548b19d58277deb600a8c = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_699ddc46ea2f40a2a787d27f12ec122a = L.marker(
                [40.577308654785156,-73.8386459350586],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_3884788d70b04d3ea1df2313de269f8e = L.marker(
                [40.69050598144531,-74.177490234375],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1b353ecbaedc41c0838a237634a3e44e = L.marker(
                [40.695167541503906,-74.17732238769531],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_00c2cfe25ffd418284cd4ec02065b693 = L.marker(
                [40.68821716308594,-74.18057250976562],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_97ca2105b2ac4d16a1a25325e522fe0a = L.marker(
                [40.68315887451172,-0.1166670024394989],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b4e8b72814384fa1bd69d281edaaf532 = L.marker(
                [40.68770980834961,-74.1817398071289],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_503fb25b28364197a4a39d0aad71ae44 = L.marker(
                [40.69314193725585,-74.1863021850586],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_09f8f9741a5940ceb9b1b52b63e14f2c = L.marker(
                [40.73257064819336,-74.28494262695312],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_5c148d07029440608808cb579e1a69bc = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8d58dd570c9642fb845ffe4beb7d02e7 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_51b5406ce76e4dd2b85adbafc14c22c9 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_7efaa598d9a142b7af77b58cf9ccc6df = L.marker(
                [40.55999755859375,-74.1709976196289],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f1d94dbc786a4ecab2f72270453ee3c4 = L.marker(
                [40.94028091430664,-73.77112579345702],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_449b7734a616486fa61db53001232a94 = L.marker(
                [40.688079833984375,-74.18332672119139],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e90d0cb60acc4c84b100fca4ec5ebdb8 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_313fd71862a04ac6886ee21dd6c09e16 = L.marker(
                [40.69447708129883,-74.17681121826173],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f3fb9ea9eb5c419e831505490b826e89 = L.marker(
                [40.687686920166016,-74.18159484863281],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1c91fad018b14b4eb7ee6e43a67984cd = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_23c8c4ae73af4ab9b4f27730e673d723 = L.marker(
                [40.918235778808594,-73.8204574584961],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_5ff5f52821db4efda229c48373c3b851 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_268b71890cdc4d2ebe06c0ba3806ecce = L.marker(
                [41.03603744506836,-73.76293182373048],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a627fde82c31482c9f9f7912f4a2c425 = L.marker(
                [40.92233657836913,-73.88905334472656],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_56c0f5502e4f44fd8527ed3d7d176fb6 = L.marker(
                [40.65829849243164,-73.58247375488281],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_13dcfde9afb24ea7982383b9820867f3 = L.marker(
                [40.6923942565918,-74.18437957763672],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_869c5ac7fa5041f2b87db15c451f3149 = L.marker(
                [40.690345764160156,-74.17758178710938],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_714b61bb88b048f8b99e64fff516bf4a = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c20c60aed82248b485a669ac127bdc54 = L.marker(
                [40.72838592529297,-73.63527679443358],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_9a0e0ccfe5bf4ffe99ecff119c32e960 = L.marker(
                [40.78579330444336,-74.16680908203125],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a66342706d4a415ea3527c89167b0d8b = L.marker(
                [40.69465255737305,-74.17689514160156],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ce535a69fd3249e89dbbbdf714f6dc8c = L.marker(
                [40.53746032714844,-74.22221374511719],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_acb6e3f89ffb4b37be30e2aeee3f652f = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ad0621eacd684b51ae30f18ed325728a = L.marker(
                [49.19465637207031,-73.41857147216798],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_2fff2f4d5a234761af16f61d4f69e728 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f638018123264428a984c80e927af192 = L.marker(
                [40.69470977783203,-74.17697143554686],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8ff4afcc2fcf459d922f4f8d72b4b94a = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_9059cd8ea6a0478a9ec005a7da50576f = L.marker(
                [40.69540023803711,-74.17754364013672],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f3b373c4c52943169a646a296e4245a1 = L.marker(
                [40.68972396850585,-74.17815399169923],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a212f7a971e14cf79c9161fa8185c270 = L.marker(
                [40.70528030395508,-74.18739318847656],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_99a96f6195c34f518d866947cae5420d = L.marker(
                [40.69474411010742,-74.17706298828125],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c4c8765c39eb4ad5837af776cd405deb = L.marker(
                [40.698020935058594,-74.30216217041014],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_476ff06ef3e243ef9bcddb9f1b981bb2 = L.marker(
                [40.72536087036133,-73.65935516357422],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8776f7e0b028486ba80646d549fb7970 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_72141ca1a7364b2d9728145e5c3e8710 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_34e61e3e4e094dabbf7450f1f9017a7b = L.marker(
                [40.690650939941406,-74.17768096923827],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_6e75f08f398f403fa0283dd659be83d4 = L.marker(
                [40.69487762451172,-74.17704772949219],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_7f71a21926ff41f08bd49c4ddab2da1b = L.marker(
                [40.7849998474121,-73.44669342041014],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_46ed2a6ef2b440ee8d7d86b8b738cd03 = L.marker(
                [40.694984436035156,-74.17703247070312],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_606f506daf424eecae178b2d987a3ef2 = L.marker(
                [40.71726608276367,-73.69288635253906],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_d4d561c976e54398a4329e6f1d218df6 = L.marker(
                [40.8537712097168,-74.175537109375],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_be393c6d00e94103802aea6f5211e42d = L.marker(
                [40.74515151977539,-73.66144561767578],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_9ee7c3d6b7af47db99629232816b2cad = L.marker(
                [40.68771362304688,-74.18171691894531],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_34538791bca9457b811bd657bf861d18 = L.marker(
                [40.68770980834961,-74.18167877197266],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c4601a804ba34c2e870e356668a5e969 = L.marker(
                [40.68777084350585,-74.18144226074219],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_4ea6d49a61a447478d6c93df19bc6fad = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_9481076b09894878b42cabe11a702594 = L.marker(
                [40.98637390136719,-74.29723358154298],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_676f34e4054647b1be1ea38cea13dcc0 = L.marker(
                [40.63445281982422,-73.6996536254883],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_34614e896a6c4a82b0189e52f879ad22 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e1cb51c9bf0846518e5e7e0927489924 = L.marker(
                [40.7648811340332,-73.53257751464845],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_86f85f6e556147e6bef132cc9db35888 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a9e1934f870b43889feaf572faa66456 = L.marker(
                [40.935455322265625,-73.83692932128906],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_5385e2c3426544ad9de27cdecb2d382e = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_517850c465af4023859baa91e11c4751 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b4ac28ecdb3945109876cc0e0c8c003f = L.marker(
                [40.57670211791992,-73.95650482177734],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a2c692cbedb449e88f023a3b90f33043 = L.marker(
                [40.69012451171875,-74.17790985107422],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_46a566a46df642169a6037f5eee210b9 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_27076891bfdf48569bf6fffcaace54b1 = L.marker(
                [40.68770599365234,-74.18209838867188],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_d221d06efd694697aca71cc32b3497bf = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c96db0305c2543c0bb07b402cbccb3e7 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a0d2e0b5b21d40f9968a1aea035e877b = L.marker(
                [40.68805313110352,-74.18333435058595],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8fbeb80ca5614c8a8fd07562419fff0f = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ace5a163067c4eb29477776f6a9398ca = L.marker(
                [40.76482772827149,-73.53260803222656],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ad5c65dff45042719bdebb9e2835f3f2 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_238e602b1b0744228865681bde296120 = L.marker(
                [40.7649040222168,-73.53266906738281],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_171bd437e21b4fbcbcc692cb038560fd = L.marker(
                [41.03625869750977,-73.77588653564453],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b421fcc263964e28846ad29086659fbb = L.marker(
                [41.03477096557617,-73.7757110595703],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f3eeddbc7989417d99acefad81511f90 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e5e3f996c7ea4385b7ef649c0483c967 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f3c22dbf4b16487fb24d9e48f179d4dd = L.marker(
                [40.68814086914063,-74.18344116210938],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_be65f308effc4998b77d6c36413c493c = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_be21e20124a547fd879823658e4c101c = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_6583f78d0f324f50bcca3dd067a28b74 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_86f41ad1b6dc4dd3b2d6fb3e455f7e2c = L.marker(
                [40.69507598876953,-74.17718505859375],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1b5ef6313217441691317c868242c80a = L.marker(
                [40.69514083862305,-74.17723083496094],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_883586d09deb4685acf394021b9c32e6 = L.marker(
                [41.016883850097656,-73.71806335449219],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_cc1cf91c96a441a98b1da6772c0da45c = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c680ba9848fd430c89a2a13a98ecdb2f = L.marker(
                [40.6877555847168,-74.18225860595702],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_6e8733095b074112ba0ef47d5d184bb5 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_648b0e6690e14a3082f92775cdebe113 = L.marker(
                [40.67257690429688,-73.68370819091797],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_7dac4f8c7b2b4eb88e76b4c94054466b = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_7af9d48b85894515bd6df737d9c65fa1 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_3f1739c5ce874440b09bf9a695d1e868 = L.marker(
                [40.6885986328125,-74.17964172363281],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_298338f7ade74ace86373d37d97f1b14 = L.marker(
                [40.69314193725585,-74.1863021850586],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_517c9966e2814d6e9b653a0ee760f00f = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_2357c2d05921404593e53c1bd87f1dd0 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_17341f5539a844ac84ab439d577890a2 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_d4785f3c3bfb4f29ae492faa85d6548e = L.marker(
                [40.69491958618164,-74.17711639404298],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ab3e1cc7518b4d66b501fe7df731324a = L.marker(
                [40.68770980834961,-74.18221282958984],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_54cd370b0a5b42edbd622cb4d3e18cdf = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_97bb74356fe24e90be6501b0bb7e1897 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b82fe1e38ce84c3db7b750d51e0a2483 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_71db31a5151f418b87db51d10c33c2db = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a5c51b9965454d3fabaa9b143a8eb1a2 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e760d8adc12246169e22dd02d7dffa5b = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_33e91390c74a4be2ba0fd88f5cef598f = L.marker(
                [40.55541229248047,-74.14158630371094],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f54dd4c4a7d449eaaac6d1531eaa9552 = L.marker(
                [40.68769073486328,-74.18158721923827],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_290451ba6e5c43b1a337ed83b8ec8b80 = L.marker(
                [40.6951904296875,-74.1774139404297],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ac8de14203fb4e8c8f484b4eee24c4e5 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_3b53f84dfb1e4cf0a4998f2097a631dd = L.marker(
                [40.687686920166016,-74.18167114257811],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_909ad09d9cbb4a3eb0bb8a2b92867d11 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_7e82c35b9bed4f558b7a545051fe1644 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_32c70d5388f245559e9205afd7f0ff15 = L.marker(
                [40.68766784667969,-74.18167114257811],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_fad02bbde9fa4f24a18d7838747770a5 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f2dd8d3db9a64d37856e55efb5770b4b = L.marker(
                [41.05059814453125,-73.77086639404298],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_94a5e620a3e4465682b0887c4e06cbf4 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_9b46017f183e45a1be10418ea55be4f2 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_19cad2b6ab1b4f20bd8eedf3dcf0fe6e = L.marker(
                [40.70806121826172,-74.17789459228516],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_7146d15586d742e7a12cf09020fdfca9 = L.marker(
                [40.69129180908203,-74.17721557617188],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a88b0c91f4f74efcba0fc91b7607ceb0 = L.marker(
                [40.64107131958008,-73.53063201904298],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_7f27c1f868334901a0af3b4a6a3ed11e = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b7e522ca8dad41caba72e34555bc29f0 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ae564d9a75d34733ad64ab7812dcb86d = L.marker(
                [40.501461029052734,-74.2450180053711],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_385bde5519d142a88ad4477ae6e15a2e = L.marker(
                [40.692161560058594,-74.17696380615234],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_802b15b24a924b369b17cda5aafef548 = L.marker(
                [40.69080352783203,-74.17737579345702],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8247b5d475cf47ad921a5689e65bb1b5 = L.marker(
                [40.69378662109375,-74.17665100097656],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_48ed96e6057348aeb2520039e08afb5c = L.marker(
                [40.73100280761719,-74.26569366455078],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c62eaf84bd154ef4915d758ec57413e3 = L.marker(
                [40.627403259277344,-74.15821075439453],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e9bfcea62086428d903f5a0363825cc7 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c09549d062a141abb6a3360bac6a0448 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ef1b786d1b644abdb806dc44af9a9e27 = L.marker(
                [40.69235610961913,-74.18062591552734],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a8b4d90248d84af5a3458e7c6c7e2bdb = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_2b2c0c8c029847c2974f05f8fc932113 = L.marker(
                [40.93332290649415,-73.76007843017578],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_477af8baaf3a452e8019f15ec3138355 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a0c678074bb84b11bcf6883a69352290 = L.marker(
                [41.15042495727539,-73.92769622802734],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_cbac7f1fb24d419ebc762c1454112035 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_dd3d80d640cc41e0aefa842a833c0117 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_6ccce14e0a2b4eddbe9326808642fab3 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_765ad9d19add4f15b7084d88d12bd644 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_97b4791ddcde40e8861bf77efb9cc503 = L.marker(
                [40.69514846801758,-74.1772232055664],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b5afc6045f8b4f68bf2a6d0285561bf2 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c621e065d62d4967a2e829965c1b6712 = L.marker(
                [40.68772506713867,-74.18175506591797],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_6e978a1ba2b042bea04ce96ab4aec531 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_5b5329d3ddd34fa993e597c157ddb57c = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f8ba41ec169e4bdfb85911b850339872 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c9ce556f672547c58ed0fd390f4ff77e = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_06f49def7e8b4cafa73d2e89cf0d1e81 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e5d2aaad99b948b08073aedd5f1d4596 = L.marker(
                [40.68784713745117,-74.18278503417969],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_cdc86e2bb1db4218b187695a380618bd = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_9c02a3a258d24ea7bbb2483500563c55 = L.marker(
                [40.68761444091797,-74.18170928955078],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_27df9d81fd9f4f54a1e682b9b3e17d64 = L.marker(
                [40.68774032592773,-74.1817855834961],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_22bb9c3491664f8da3936964282af1cd = L.marker(
                [40.26865005493164,-74.0261459350586],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_19ef493eac2340099adcdcbca8c703e5 = L.marker(
                [40.62735366821289,-74.15422058105469],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_285a1791edcd45298e66747a4f68b57a = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_9b473b9cf3ad4a97a0a08b990b4c2f2b = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_92e3a3c30f074ba1ab60af2b4bec167c = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f3f8d6e1c839461a9125a98639926ba8 = L.marker(
                [40.68778610229492,-74.18276977539062],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_d0d18973b3c949a09451f1193dbd4e90 = L.marker(
                [40.6909408569336,-74.17731475830078],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f70c8fab810143948db3ec20847a5ab5 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_5f40d1767ea14bccaf0c202b419f520d = L.marker(
                [40.69314193725585,-74.1863021850586],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1a3debc530264be98ee858d6682e6fec = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_3c01eb21d1d44961a8b00a2c01a2a848 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1ee7b373886646f397b0e61ce0cff7d8 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_2b4041f913bc4a859e273ec2157f5e3a = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e9678173c4354469af83c653dded89e7 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c1e6c2c6271a4b8c951fdb442ccd621f = L.marker(
                [41.05612564086913,-73.91979217529298],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e42ec03090f84c4ca59c24f5a4ac8ab4 = L.marker(
                [40.68849563598633,-74.17987060546875],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_35e406d0c0434d82a98d232efb9c4a57 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e37c6405452645b4b475fdd468a8bded = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_5e1269e67b7a42b3b9b8c7d706b57142 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c703cbc60a8c40728c3fff5f39e7d059 = L.marker(
                [40.687721252441406,-74.18173217773438],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_7aa5bc9d3ac34f19aa477f9558951a38 = L.marker(
                [40.69038009643555,-74.17762756347656],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_03d49ea07d874e759d0b510527b2dac1 = L.marker(
                [40.53280258178711,-74.22219848632812],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_4c012e3abb41409d981a6d05c0922004 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_38143737ad464eacb025aa81845a0c8a = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_9b41165a80c74685ab607b3486c2f8d7 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_6254880c181e4c56b74ef12dcdf6c396 = L.marker(
                [40.69471740722656,-74.17694854736328],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_9687b21478594520bee4b2d9307f56b5 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_7519c9d145fd46228d91d187028b1a07 = L.marker(
                [40.57572555541992,-73.96342468261719],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_2046ba52563f4c178229ee1b933145ec = L.marker(
                [40.77644729614258,-73.56111145019531],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_64d9ba71e74148df98c57191bcad20b7 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_decd3d2d095245fabf29172546b45af4 = L.marker(
                [40.69523239135742,-74.17746734619139],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_343f6291d3eb4f15bd98eb1bd27d004a = L.marker(
                [40.71036911010742,-74.15693664550781],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e8760b1b68854249b8e788b747507372 = L.marker(
                [40.69548034667969,-74.17771911621094],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_3f050649d58f42b08bf9375c28cdbf0b = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_730510e6d1f945f28677b7ef6bcec910 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_606f958f26264a218305b6c9d285d9d0 = L.marker(
                [40.575210571289055,-73.9985122680664],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_d1f03eb356014e7cae7852a4ba2c56e9 = L.marker(
                [40.687721252441406,-74.18177032470702],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b7c7caede5fd496a9467232fd27629b0 = L.marker(
                [40.79316711425781,-73.6805648803711],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_251c23a38c2d484cbcc250cc3027ef92 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f44da16ce4274f449ce8028acca61a9f = L.marker(
                [40.94786071777344,-73.83102416992188],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_043466fadb584e4e9f9a1b5251eea193 = L.marker(
                [40.74600601196289,-73.69267272949219],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_651ab0b6fd50463c9ff15dca4b4c938c = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_4474d456758f48cdb43ce3aee54f9d2e = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1a6fe9b9306748adbc6923c8bdc8fb3c = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_20ed9212e716406a897522f8e6599e17 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b8930f7d85f14d5ea321df669b26b235 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_2c09d0b79ccc44db91e97f6ace37f101 = L.marker(
                [40.838035583496094,-74.19275665283203],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c23f0ced48944d8fb95037f31e8e6339 = L.marker(
                [40.69554138183594,-74.1778564453125],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_d6069ccdef364f10b17218506ba0ad7c = L.marker(
                [40.68777084350585,-74.18171691894531],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_fe805e12186b4085a602238a114c8f4a = L.marker(
                [40.69517517089844,-74.17742919921875],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e0b2c8078ab744e58348bb820378e1aa = L.marker(
                [40.69494247436523,-74.17705535888672],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b49339a70f60489abcdb3f981c3f8c8d = L.marker(
                [40.69017028808594,-74.17781829833984],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_d27549d8b4ab449599219a20851ea53a = L.marker(
                [40.69411087036133,-74.17671203613281],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_4fa862a36e2848c3bb4e45c38709275b = L.marker(
                [40.97608184814453,-73.80429077148438],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_904da269587c4c8fba9cbaa112d2e4c9 = L.marker(
                [40.68992614746094,-74.18407440185547],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b399b90868e04c93821649b5369c21a7 = L.marker(
                [40.69506072998047,-74.17720794677734],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_61105031069340d4a1d9eb5b797aed00 = L.marker(
                [40.695411682128906,-74.17778778076173],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8b17713c47dc4f5494cb900596cba523 = L.marker(
                [40.507408142089844,-74.37515258789062],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ff1d4b1eacd84e69a1c4cb8e9b0bcff6 = L.marker(
                [40.72394180297852,-73.5876693725586],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_562242436494490b966f4e043cbf3d32 = L.marker(
                [40.693862915039055,-74.17667388916014],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b4b9be92535d4c8fa9fcdd521823e484 = L.marker(
                [40.662357330322266,-74.17704772949219],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_41fa7f3ac53740cc8d2a04cdcacf638d = L.marker(
                [40.887489318847656,-73.61089324951173],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_5113e5645e4040f79fecd3f70982aa89 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_5931e6fcc2044bee9e2fcfed1103ed21 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_9621c0a6ff3c43e294714541c3f09145 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_be9036f4382a45e2b6f755a07e19eb2b = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ef51dabf0cf1433c8282d6327e66b18e = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_6b86de4f72544ea698b7c3d3868faa33 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1606a836aa88420fa24825f6367a8cb3 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_463a6a94b83e4c2fb1daf2c85faae13c = L.marker(
                [40.69054794311523,-74.17753601074219],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_9c35eece6a5c4a21b2e163cae63f8558 = L.marker(
                [41.024280548095696,-73.86818695068358],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_7abb7f736eb5486a852b9509d020c174 = L.marker(
                [40.687782287597656,-74.18267822265625],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_24f05401c6ed4630b84f80bac0770529 = L.marker(
                [40.68772506713867,-74.18186950683594],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ff0dff067e634b7b80a202615d2b82ea = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_0fb9f7ee3cd74cc1b8435c44d95949d3 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_563d3cf2498a45cab10c85782e460699 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1e5509c4b5864b51b7ffcfb56c494488 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f36d1d54b78e478cb4f55ba2d3b118e7 = L.marker(
                [40.692039489746094,-74.18130493164062],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_fc3af61cba924f04b60e2e308efce8a8 = L.marker(
                [40.55462646484375,-74.19615936279298],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_af60706e6df946479118d384a3b2b312 = L.marker(
                [40.575870513916016,-73.9767074584961],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_0da4e6c774d14de39c442f86960218bc = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b6e780d633664fe7b0c90d1cfa8ec8a5 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1c66105274734ae397c03ef9e66e229c = L.marker(
                [40.687686920166016,-74.18156433105469],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ffb57b20c5404b27824eff1c9c7ca6f9 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_bbb6bd6bbfc4460ca27e018cbe1f332a = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f9bf39e1254e4049934e81a352b649fe = L.marker(
                [40.694034576416016,-74.17670440673827],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_0029de66742f408596485d4252965578 = L.marker(
                [40.68769836425781,-74.18148803710938],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a87477740ad444a9bf11e98504fdc21d = L.marker(
                [40.6906623840332,-74.17754364013672],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a5f0c0c7f8c34e188c84d32709212867 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c81de10d577345ae8a9039ec807e54a2 = L.marker(
                [40.69314193725585,-74.1863021850586],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_88ab3b02ed33466690d3ef9d090b88fb = L.marker(
                [40.695068359375,-74.17717742919923],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_95613d1f74604bd9bdf00d657a679204 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_6eced6f333a44097ac48214dd0972f0e = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_9fbb1d56cd7c4fe88d75c780fcb6bcee = L.marker(
                [40.98324966430664,-74.16829681396483],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_9063e1c073ea45919b686b15e25ede89 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_60f9bcda121d43ffb5cf4b442516754f = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_118c467b5dc44b9da1517922096b4d27 = L.marker(
                [41.08624649047852,-74.1566848754883],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_d47d513fd13b420b8a3809ef3256b049 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_20323869fc5a4ca798b1f592ea345072 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e4d4014f74d443739aa9fb61987c4255 = L.marker(
                [40.67060852050781,-73.62162017822266],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a37a1f618e804ebbac689e65a2c7c4f7 = L.marker(
                [40.77879333496094,-73.4280014038086],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_2dd51bbfe877456e934a19ebda88f738 = L.marker(
                [40.68770980834961,-74.18231964111328],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_db99837d10d74e40b44592ac9cc7cb24 = L.marker(
                [40.60417175292969,-74.22815704345702],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_7a6c82cb711e47ad8c91d909010e33f3 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_37f6beae12a34231969a6d418973969b = L.marker(
                [40.69068908691406,-74.17747497558594],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_9761372c37a245a7b822a0ca658b4a8c = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c83d9908d2d44b0a8840afccd17aee8a = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_561f77688e824439abc2f64a75fa44e4 = L.marker(
                [40.68770599365234,-74.18183135986328],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b542e8c141b5462fb824b72aa2fe336b = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_6028af21949e42aa90606bbeb962151f = L.marker(
                [40.57539367675781,-73.96842956542969],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_dd2c2042385a4dcbba8e11bb91e2516b = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8b28deacab79406c9bce2b6c177f1ba5 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_0c19dc342e494200a22f23f38dae1e5b = L.marker(
                [40.69485473632813,-74.1771926879883],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b128ad33151d4a2f8d4fc7c4a0bec8b7 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_24d8af351d494b8c8d3aff80a694df56 = L.marker(
                [40.732566833496094,-74.17402648925781],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f65ccc13c840488ba3c5a8331333330d = L.marker(
                [40.92459487915039,-73.8390884399414],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_d9be295f317a4b6d9bff78865fbb8821 = L.marker(
                [40.777168273925774,-73.67108154296875],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8252c493e7ce474398d6d717da574bc5 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f3a3fe2da799427aaad26e3397adaa78 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_75c9564fa17345eaa9254d4bf9aafe7c = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_15b45e45f8a24d858e0be208a262385f = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_21eef96bd1b64caeb6113df8382dd78c = L.marker(
                [40.98798370361328,-74.2346420288086],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_fbccdda87a8a4399a5632339905c1517 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b167153f2ed442019b8f35e452717746 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1dd7947df20f4a9084cfa03438214f9a = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_332e3043c81a4f3e9051c5b0d11b8a7a = L.marker(
                [40.69495010375977,-74.17704772949219],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_9d4947956bab4dcaae3db4cfd1e7c751 = L.marker(
                [40.76039123535156,-73.6319808959961],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_6926bd3ce2f84c50b6358648f2ab94e2 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_52867ccddfad4581bbddfa54c0435ad3 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_17af1f5698794221930d29ede31dadab = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a780ec8af9ee41aa8fdf1cc416804b45 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_6f5bde992bf045f38519125271e55813 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_aac497e05af640548e805e787374dccf = L.marker(
                [41.057289123535156,-74.04696655273438],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_bd1259490e684558ab4ee99733b478b0 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_822ea03bcc4044aaa783eaedc1061515 = L.marker(
                [42.30679702758789,-57.38808059692383],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_7d2541b78906402593d7e98b770c7047 = L.marker(
                [40.689956665039055,-74.17811584472656],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_23a08b90ec584b87862b77591bd19c7f = L.marker(
                [40.695213317871094,-74.17746734619139],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c8a05e936c88461ea4fedb2e9eaaf96a = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e74145f2ffbb47a783ab270fc8724ef8 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_0b3e500ac8274db49a63d1fe1fea7017 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_99d807652a674ad5b01c533f41bf5126 = L.marker(
                [40.69047546386719,-74.17761993408203],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_2cb984f6c3d447cb956d6566afc0371c = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a5e9eb23b6e342dcae6f16c6b3d6fc28 = L.marker(
                [40.69285202026367,-74.17678070068358],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_7d3c2d2e3b4549c9a4e7d44d37ae5cd3 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a88122ff4ec94c64b6e2757f5c18a7f4 = L.marker(
                [40.694202423095696,-74.17671966552734],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ada8d05650b5493f855ad858c09acbca = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_38edb7ccb05e43d4898d93f657bf5262 = L.marker(
                [40.68777084350585,-74.18260955810547],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_6a6d27c90f41425f8ba7b52c43d00944 = L.marker(
                [40.68838882446289,-74.18370819091797],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_bdd76e58ad274ea6a04b18a7c8115030 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8dab8c4f9f0442379e35a5c1ea91e7da = L.marker(
                [40.74711990356445,-73.52417755126955],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_fe9a4777bb12430bb1e87d00176feda9 = L.marker(
                [40.69314193725585,-74.1863021850586],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_0baa10ff44a747729ab15f475e4ca3c5 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_9cfd09defbab4ac7ac8ba46c3076bcde = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_2c691d5e58924a8eab2f8f1b0438ac49 = L.marker(
                [40.68785095214844,-74.18299102783203],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c06ff435782f4bd889e8c8aab98218da = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_de8ef485e77b4e4ca683f13d716b915c = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_628a445e1ac1456d93063df29f8fac98 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e70877d5a57a43258e8b825e3113f787 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_7ed173137c9a4f20a9d519d04902e5db = L.marker(
                [40.6896858215332,-74.17860412597656],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_157416567cd842cd954de5556a5a39e6 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_200ec0a6f1df43d3adc444e46a5e7e67 = L.marker(
                [41.05142593383789,-73.53506469726562],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f92a841ee0e34f499dd3cd81e0b584a9 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ba1dfa90b11b4c2b96e96fd58ba0cf9e = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_4baea82a13b947afbc2935de3c006ee6 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_16f84bc5926d47b1a05ce03409473c44 = L.marker(
                [40.80564880371094,-73.48265838623048],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_bdde3b0463004416bd5e6f512c0f5a44 = L.marker(
                [40.694988250732415,-74.17713165283203],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_02ae37215efb40ab82268f77d652c2ad = L.marker(
                [40.919864654541016,-73.8643035888672],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_25ef62115ca849eab973019526516ca8 = L.marker(
                [40.69314193725585,-74.1863021850586],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_6a169cea6ca846b8aebb0cc69493d973 = L.marker(
                [40.54436111450195,-74.1583023071289],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_2d87d5b94d4a4dd3b220ebe20431eddd = L.marker(
                [40.69055938720703,-74.17755126953125],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_703d5610088848f09f24228a23522c1c = L.marker(
                [40.6954345703125,-74.17765808105469],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_38ca1cdc12304c0bacd3fee95fa143e0 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_357a328aa8644bcabeb75820dcae9430 = L.marker(
                [41.150405883789055,-73.92771911621094],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_2f3c6a6d754444fe9bf91e1fdd62b2dd = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a4bd64b329864f5ca6465b3207a8ebbe = L.marker(
                [40.7673225402832,-73.69229888916014],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ef3497da673d44cabb7205526713b619 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c30fd57369bc4456b5303e822d462fe7 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_7bac485997354c91b1dc69078ae32db2 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_3f19573dd8364419bbdd31cc1fc23cda = L.marker(
                [40.687686920166016,-74.18144226074219],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_df9eb2859aa34842b98e13c0663df8db = L.marker(
                [40.69474411010742,-74.17693328857422],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c42bc511f8b044e5b04af22c08ec26e1 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_379157c2dc764a5b9b572384de8d499d = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e9fd8428008e46a0b4600e95db74fb9e = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_7b311403cc5f4f7ebf9520751aeb7098 = L.marker(
                [40.57731628417969,-73.9629898071289],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_396536ce4c884e4d93922e99e22acd56 = L.marker(
                [40.704734802246094,-73.62335205078125],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_6c78f704ee874b47a2842c6abeea8831 = L.marker(
                [40.56898880004883,-73.8617935180664],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_58d7746ddb8a4589bf44644bf4c442fa = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_192b6455f16f4d3c9c2b1275130a4754 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_fc8e11c586164d41a7a8c81f2cecc9e5 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1b4f2d5e12f24bc1a0dd386f0bc8876a = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_dcc728235ab24da58fd4f29532f256a1 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_65c9c55060ab46e6b1163f82990b07c8 = L.marker(
                [41.06896209716797,-73.70413970947266],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_5303d025de8d407daf430a18dbfbd67f = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e150883486684885a5d195c34dab04d2 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_3e87de76124f48abbc525d4a6c30c3b4 = L.marker(
                [40.69065475463867,-74.17753601074219],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_53a9daf61f5744629b90be406e91adcb = L.marker(
                [40.92009353637695,-73.86534118652342],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_05357288927047beb3a6d8c43508befe = L.marker(
                [40.941307067871094,-74.12061309814453],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_520cfaeccf4540f8937b66a1c0a371c6 = L.marker(
                [40.69395065307617,-74.17684173583984],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8549629711ed42049588e6e10bbccadf = L.marker(
                [40.69577026367188,-74.17909240722656],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_0040007ec1f3405daf63a14e41855b62 = L.marker(
                [41.03440856933594,-73.76902770996094],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_2d2d54f37daf4a3285a96decc885cff0 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a6e326a7baa9461c9d2488d14780c9a8 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_738f6505f91b4d7e991bb37921015b76 = L.marker(
                [40.69066619873047,-74.17755889892578],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_77246dfc53134ecb9f9ea6005bff1920 = L.marker(
                [40.95989608764648,-73.89319610595702],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_080330fd06054e0ea45b96868d782cf0 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_73f42f3d0b6842d88dd37ed1e9e3573b = L.marker(
                [40.69206619262695,-74.1813201904297],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f4ef5d8739cd4c70852a5fcddf2499b5 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b8e597572f7046178203ba8fc0396a82 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8080ea2754ad435a93bd8f5dbde42dbf = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b0a7b1936f5641f3bebcb26e4d69db55 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b43d6c8930a640db82cb059104e02ac9 = L.marker(
                [40.68772888183594,-74.18138122558594],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_abd4b3fd33b24bb3a286b0b71c700987 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a80eb5889cd34448b1f620fb263b3193 = L.marker(
                [40.69332885742188,-74.18309020996094],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a31037790ac4427b9660c16b87c056b3 = L.marker(
                [40.46578598022461,-74.4590835571289],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8976fe2cdb994360b151679693fba7cd = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_25034882f23e44b08d03d10a612f5d11 = L.marker(
                [40.69538879394531,-74.1884536743164],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a7ab602f749f4e7c9f3072471b02fa8f = L.marker(
                [40.69111251831055,-74.17730712890625],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_504721c96fe6459290a1dbaa89a90715 = L.marker(
                [40.93073654174805,-73.90029907226562],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_34052553601a468c8d695a52614f85ee = L.marker(
                [40.69017028808594,-74.17798614501955],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_430d8efd83bc4d6e807a5d55e3334127 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_4068a67105f74870b5b01642a7a55bb2 = L.marker(
                [40.687931060791016,-74.1830520629883],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b03632e9fe904437ac34902935501828 = L.marker(
                [40.55322647094727,-74.0496826171875],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b7352d907d3345f1885364fcb536af5d = L.marker(
                [40.356460571289055,-74.64940643310547],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b1af6f26606342b0bb8eff5d9b65f063 = L.marker(
                [40.860687255859375,-73.4019546508789],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_6ad5149057974a33abea1df8f15d4a33 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_3707bb38a9d54057a9cab92ca9813f7c = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_d532d2bdd61c4d069bac9e4dff7896c9 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ac89cab393f040efb4e1cde375117df0 = L.marker(
                [40.82496643066406,-74.20677185058594],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_217cac9459024cb5a26061dc03d48a48 = L.marker(
                [40.69017791748047,-74.17800903320312],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_7fc3971100444b52aa7669fb22340acd = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_9545197ae1a14895917fb586204a7341 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_5e913337b0b142e793d4d6798355cc9a = L.marker(
                [40.69025802612305,-74.17770385742188],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_fefc2085b793481eabf8b44104b2aa8c = L.marker(
                [40.68793869018555,-74.18297576904298],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ed356a1ad00c44d5abbce6f0a0179681 = L.marker(
                [40.57534790039063,-73.96280670166014],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_9a95eb7ea85e4920910b43a714166301 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_39fb4de4ad724b21b08e5ffdb905028e = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_36e2f3856c6e4b9a8bdcec96f892d88b = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_36ee8932e31d44b4b8623dc9c8e81621 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_be701df4c1ae43b482cf7af45ba8fa42 = L.marker(
                [40.57388687133789,-74.091552734375],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_98f13e055ad545568a601be23634186b = L.marker(
                [40.68936538696289,-74.1785888671875],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_bf75fc57117d4865b2e6570e898d98b5 = L.marker(
                [40.69143676757813,-74.1770248413086],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_82d34b14f3d74e5f9d89fbcae591c781 = L.marker(
                [40.630348205566406,-74.15252685546875],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_2a461749f89244f6911cd7b649752650 = L.marker(
                [40.69464111328125,-74.17686462402342],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_18901886be5a47c897ea086f3fbeed9c = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a057befc4f28434881e5e9a90e0403f1 = L.marker(
                [40.69006729125977,-74.17786407470702],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1bacd8694fb04cb186d2e574bd3f5ce5 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c81a82ebbf2041c5ab47b5038a04442e = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_fb31d97cb58946b096f46a85a5e58687 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_21c81b37512f471cad802f69260cf020 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_712212bcda94478489d195a9c707f243 = L.marker(
                [40.92623519897461,-74.0281524658203],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c1231713749c401a99d6d70710ce9500 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f15b437ead774b1685e1a91427ab44d0 = L.marker(
                [41.04612731933594,-73.68427276611328],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_125884a3245a423d964677bf5adc1b68 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_697f2f3c0ac644a8978f86ad2fd12e6c = L.marker(
                [40.76484298706055,-73.53263092041014],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_50f18416f4db4cd98ee512a37447ca73 = L.marker(
                [40.68771362304688,-74.18159484863281],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_748008255218474985afecfb5cdd6ee9 = L.marker(
                [40.68767547607422,-74.18232727050781],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_884eecb88d3942df8cd193b0a45a5b5b = L.marker(
                [40.58894729614258,-74.16705322265625],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_2cf2c207540a420590b61f0930746cb4 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_98a908ef81784b01a7ef5b5b5f279b4c = L.marker(
                [40.690956115722656,-74.18113708496094],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_47138177e00f4534a1edea04c7c3456a = L.marker(
                [40.69510269165039,-74.17720794677734],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_81ef2cd28ffa4d78b87f4bda63f3c3f4 = L.marker(
                [40.68902206420898,-74.18416595458984],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_5cb71c6b01364916a9860aaf3c914f0e = L.marker(
                [40.69501876831055,-74.17710876464844],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ce33f6840777411090ff98b39bb76c11 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_029e732d4b3a495f917e5716062973b6 = L.marker(
                [40.69504928588867,-74.17713165283203],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ee8bad8a15d94934b2188d323d95316b = L.marker(
                [40.69535446166992,-74.17755126953125],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e4b61d7c0dba4f9da852284ca26cc32b = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_61e359513e624632bdb85550cd665eae = L.marker(
                [40.56658935546875,-73.88675689697266],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8b6703d4125346c7b4699a006b045fee = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e9fc1dc67d44498d96e7d7f5373314dc = L.marker(
                [40.576484680175774,-73.96308135986328],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_6117103a84f54c58be4900087fd185ea = L.marker(
                [40.6883544921875,-74.18335723876955],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_cf22957ba07641d3b02f4b635ac72580 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_290bac6e5f214c4b99483010afca334b = L.marker(
                [40.68621826171875,-73.66709899902342],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_7d6f77808f9549bc8a49e4c98357a6cf = L.marker(
                [40.58418273925781,-74.16560363769531],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_30ba844b3aef4905acacba3d36a746ca = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e4fc1885885d45a18ee6cbb1092a5e2d = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_bf071752054e4f30879094d8f5f7b487 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_17b5ee2c6baa44549582f29e90f052ce = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b9efb84b639b4104bab942089f626c6a = L.marker(
                [40.92018127441406,-73.86723327636719],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8cdebf418f1841caab9f129bfb63b086 = L.marker(
                [40.722564697265625,-73.69056701660156],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_de6b916acd5d408eaf0f80e2c5653207 = L.marker(
                [40.74212265014648,-73.59044647216798],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_4618fd6de55b485fb2dd3c91b7978555 = L.marker(
                [40.68808364868164,-74.18330383300781],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1e94646f6f00440ea0ccf3e29df92d96 = L.marker(
                [40.69112777709961,-74.1771469116211],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e2cc414de0c24f379eaff45c2ad64844 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_05b58e3495914e2cb79e2ced78168d0d = L.marker(
                [40.68769073486328,-74.1816864013672],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e626a456febf4770b8e2a0452e3bc8c1 = L.marker(
                [40.68774032592773,-74.1817626953125],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_39345c9d241e4bf0b48e6183614efa87 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c61c2a4f7c814688974ef4d9d722e5f4 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_2677a2ec85d84201b9a6836861bb3883 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_53c91b3ae58c4f43afe5ed7ffc5707aa = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_012140b9ebf04be68306764f8e361e21 = L.marker(
                [40.69515228271485,-74.17726135253906],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e707b30b92814d9eb965029a989e0b27 = L.marker(
                [40.694820404052734,-74.17726135253906],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8997ec66036e444e8feb7f9d152b2c1a = L.marker(
                [40.74606323242188,-73.58989715576173],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_5b52607abef145f987292d40e34e69f4 = L.marker(
                [40.94343948364258,-73.83707427978516],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b67bda4472584c91893ef148195016a7 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_25220f6d9536400d884e88d5824ee484 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_abb5572c88264d919198c549d48c4917 = L.marker(
                [40.783836364746094,-73.53302001953125],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b059aff8285a4474b566248069060ba3 = L.marker(
                [41.01707458496094,-73.87386322021484],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_d94146d497134e23b53ee18ee8e0a691 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_9d06aba6b35544f8a110c8b1722913a6 = L.marker(
                [40.66234970092773,-74.17704010009766],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_902c224bd2d5456a8e505096bc8fe557 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_0444cd6c70e9467f9f0ab92b37589239 = L.marker(
                [40.690101623535156,-74.177978515625],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1f7c9e5e6669498ca4c4cd1ab68147d7 = L.marker(
                [40.6895866394043,-74.1783676147461],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_bd0f975ada4f456e8817b061facaeead = L.marker(
                [40.6954231262207,-74.17750549316406],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_95f843b0e7954a5f94daf26418289634 = L.marker(
                [40.69075012207031,-74.17735290527344],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c152dbe87d094660959ce68b380392f9 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1d322a2a9aa2433f89cfb526ba603466 = L.marker(
                [40.69470977783203,-74.1768798828125],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_3f9beb76baac4244bdf1610e74bf1db3 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_9923c90c5f0743edb9c3fe62a7607912 = L.marker(
                [40.94683837890625,-73.8665313720703],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_aeff323b58a24ce486a08c6ec06cad91 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e5ee9ee852a54209914056745daff11b = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_9d9c815022b442e69d5f3593118c8a19 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a2aec9cae9fd45a28dfa1d762971d2de = L.marker(
                [40.57649612426758,-73.93881225585938],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_aa789320b40e4e0caeba8bf0412539a6 = L.marker(
                [40.69491958618164,-74.17700958251955],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_0b9498ce60164acbb6d19773cdfd9fd6 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_d8d0ec0ef07d4a74819874bed54a86bf = L.marker(
                [41.05447387695313,-74.12957000732422],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_9be6c42246594fd6abec44375aaba3b3 = L.marker(
                [40.574867248535156,-73.99958038330078],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f5fe4799967f4744b3d34661a8581b31 = L.marker(
                [40.69050598144531,-74.17754364013672],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_6590acf1de614a478be9a87637b8952c = L.marker(
                [40.65027618408203,-73.6438217163086],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8843e554707044b4b2ed0c2c065fad1a = L.marker(
                [40.73507308959961,-73.59437561035155],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_481a192247ba48c6af7c329a36aa81ba = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_3232be2ff7ac439ba5bc7588e8abb57c = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_11e765a13bb648328845c882b8126506 = L.marker(
                [40.69100570678711,-74.17731475830078],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f7503138a60649c987f2c21a65c3ea86 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_15d34c1653934cc0b53bfacb78f5ea3e = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_3664b2a2e92c4edfaaff213b40674a99 = L.marker(
                [40.6912956237793,-73.6488037109375],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_66b002324cf14307ba115378026dee77 = L.marker(
                [40.690105438232415,-74.17790222167969],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_eaa2c119a4434bb29097ef4106dbeeb3 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_336d4a30129945ae917270c628b4dba3 = L.marker(
                [40.95875930786133,-73.84185791015625],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_756ad5f2bae542a3b00b7991c519d1b5 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_2eddf9abb0274dcba9f37795397618e8 = L.marker(
                [40.69463348388672,-74.17689514160156],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_0f4ff99175804742bd402c9c0bf4c38d = L.marker(
                [40.68771743774415,-74.18233489990234],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_2073d80f4ee44c03a90fa0f87dcdecbb = L.marker(
                [40.82429885864258,-74.25206756591797],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_0075034dfbcc4b90a5ecfe90299b0b54 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_43049304fa19430b91d7ab023a57e23a = L.marker(
                [41.15342712402344,-73.67224884033203],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e90cf1593f0049f6bf69e89da4be820f = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f0b490773a074c3797c60bc0d9653dc3 = L.marker(
                [40.68762969970703,-74.18159484863281],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f4f43bd09fd949e994f7fc32c2bb391e = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_d0e773b137ee4572a31983661256ff2e = L.marker(
                [40.64751434326172,-73.67806243896483],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_289142298c5246ee9eb6368b925c2e03 = L.marker(
                [40.69117736816406,-74.17709350585938],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_2f48a46b5c894681858b48ea345a121a = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f8d7b35428f349599178074d2a00c04e = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e6773a441a744c499089e42b706c281d = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ab69b7ca28594d2395bebfd1750dafe0 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_92cafdf1930e4a50a1e3877613b61de1 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_14907763f4394d598c0140788f5f0417 = L.marker(
                [40.66076278686523,-73.65980529785155],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8655f665393241b28f46087f7f5b5245 = L.marker(
                [40.59343719482422,-74.24896240234375],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_586271f7bc33423a99301b430e5ad648 = L.marker(
                [18.62594413757324,-76.65936279296875],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c83773d2d5d54da4b49cc1c3f2b8c1c1 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_42cab28a6123455188bdd29924278246 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_4525961568e44601a469921028eae5ff = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_83432e9256884f0790fec9d165787986 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_7cfd9badd139474ea0737c8bd3ab1e54 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_85889ec9d4de44668beedafdebad6536 = L.marker(
                [40.694969177246094,-74.17711639404298],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_49ee6992f7894bb98e7fcd86369b1d97 = L.marker(
                [40.93931579589844,-73.81717681884766],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_5753ff7b1a3e4589bfedc07878e2fa73 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ba1d571384264a01aabb17a0ac0fbfc9 = L.marker(
                [40.69200897216797,-74.18135833740234],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a40384251b4e4864808500ad633b8c6c = L.marker(
                [40.57353210449219,-73.85955047607422],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_d175ec680f5f40fcac077528f2d704e2 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_7bf2f65ea2f040a28ab85e2af0ce3f49 = L.marker(
                [40.56746292114258,-74.32971954345702],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_400e89db92834e3392a232d0837d7599 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_29363fa6dc424376922fcd0c30a02af9 = L.marker(
                [40.8028678894043,-73.6649169921875],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f67a364bf69341f390d6be0dbdfb0c43 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_bbdecc49d21f4ab68b1a552fe22f3c03 = L.marker(
                [40.69503021240234,-74.1771240234375],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_575b101d239e48eaab10c3a8b4cb26eb = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_6567194adeb74f96adb0798a4af72837 = L.marker(
                [40.69064712524415,-74.18756103515625],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1e6c981f393a4bc9bae6b6e9c2446cd9 = L.marker(
                [40.93130111694336,-73.84014129638672],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_874578f7f112448285f59c79910953e2 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_33dc3856875a4f699e12cedf0c19d283 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_4e0d8cdb9559441b8b1d1bfd38ee1fc5 = L.marker(
                [40.97996520996094,-73.68616485595702],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b25e3b7fd5fc4790b42a5f254aa744d3 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_897575463ea24876ba772a424ebb0a96 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_d38644ea76e14222ad79b1ad043cfbaf = L.marker(
                [40.9200325012207,-73.752685546875],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ecbce42d3d784fe8a44122819cb21767 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_7c97ce05dd8c4bf6b9eec5665e061135 = L.marker(
                [40.69514846801758,-74.17729949951173],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_cf930e53cfea47b6ad9c9c6c51f253d6 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a1520014157244b88d613394f6e559ee = L.marker(
                [40.69048309326172,-74.17754364013672],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_3ce5ef08808a41d4a2f9242916f0070f = L.marker(
                [40.69218444824219,-74.18110656738281],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_079c53cb15ac46208c544800569aa05a = L.marker(
                [40.69075012207031,-74.17749786376955],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_02eccfb856164b0290dbae08c0f585bc = L.marker(
                [40.69487380981445,-74.17706298828125],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_d5c6b6d72b4b417b91d496f1ddd396de = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c6f0078494f746f8a589b469a8101a4b = L.marker(
                [40.68883514404297,-73.6988525390625],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_7bf4451081bd4348900fc5e230946816 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_baac843c74fe49158589b0dc58a1b315 = L.marker(
                [40.91968536376953,-74.08427429199219],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_d5ed68c909b048dd815530d96b8aa7c2 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a164b77f5ab54ade896463737d078c64 = L.marker(
                [40.6949577331543,-74.17717742919923],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_02bd15474bce470b9da79207aa8a24ef = L.marker(
                [40.69408798217773,-74.17686462402342],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1a98778ede22447cbad576f36b26f3f6 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_d73edf66d4c449ddbeebbd137f847c6b = L.marker(
                [40.7871322631836,-73.49465942382812],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_6621056b163e411081361d200775d3c5 = L.marker(
                [40.687835693359375,-74.18267822265625],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_2ee69758eca9401e805de16a7ea6374a = L.marker(
                [40.69543838500977,-74.17786407470702],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1e0167b9464e47abbaf690ffac492bba = L.marker(
                [35.562171936035156,-74.23307800292969],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_5685aa4a74ac49a9872bccf899b53510 = L.marker(
                [40.94351577758789,-73.83309173583984],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ce88fcfef7a943539ac37bbde305f66e = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f5d23a0b60994565ad4778a74e2b7acb = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ed0eb4cd169a42f8b90cb3ec85cb0218 = L.marker(
                [40.69497299194336,-74.17704772949219],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1d6fc27ec57745e19144f333df3ae8cb = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_bf0063cde2994547baa34aa071b3ed29 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a6805749b0294d29849fdb24a531e414 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_6dcec798faff458dbc226ab6b4eacce2 = L.marker(
                [40.57621383666992,-74.11618041992188],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_07b97ff9e4ec46469f5f04779bb27a61 = L.marker(
                [40.79991912841797,-72.77095794677734],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_dd99a89e5d02465ba5a4277447b1e1a5 = L.marker(
                [40.68772888183594,-74.18253326416014],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a092382c9e3c423c89d0b6e175fd9a3b = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_3effac6ce6434c939918eab24f5dcd8b = L.marker(
                [40.57625198364258,-73.96216583251955],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_0f99fb8fbb18465284978dfa95845faa = L.marker(
                [40.74232864379883,-73.68952941894531],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_4e4ff670c15f41a3926001ac8bd18fae = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_12932a20680042e8804c964d93ce219a = L.marker(
                [40.68772888183594,-74.1817855834961],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_5da70332ba5d4aa99633e5be068d69c4 = L.marker(
                [40.69314193725585,-74.1863021850586],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_881391ed04c542edbab8a30dc6a15c46 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ef68da2f1e3246fe973595d59ba98788 = L.marker(
                [40.69548034667969,-74.17765045166014],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_92f2c53a104648fdb128625b59785f0e = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8cc2decfd7b5434492557db297298bf6 = L.marker(
                [40.11590194702149,-72.44942474365234],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_97208510da2e4f8abf671a7b6900e92a = L.marker(
                [40.621971130371094,-74.26590728759766],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_97a266b72f41453f836ef11387a790f3 = L.marker(
                [40.688663482666016,-74.18394470214844],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_3d01967f179441f382fa8fe0b79e6ccd = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e5c6994fcf7c4f098af5d90b28ef895f = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8c887fa12e63488fad264a47986ba184 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_d8fa545755dc42eea4ce7746064decf3 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_eef03c82193d440183e28cf7a352dd82 = L.marker(
                [40.06171798706055,-73.34426879882812],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_70f53a61398849f293b2b3b3343d3840 = L.marker(
                [40.69543838500977,-74.17765808105469],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_d7affb49dc824d0abd74f3be53887791 = L.marker(
                [40.57635498046875,-73.95558929443358],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_7ed9defcf9c24290ad6a7abf42b42832 = L.marker(
                [40.69502258300781,-74.17711639404298],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_62b3354092e74b40bfce246ecee2b85b = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_dcde61db3d9e4ff584c7e0974c4521c1 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1b93e51d617744b09fe27dcc1185ac78 = L.marker(
                [40.75894165039063,-73.66965484619139],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_96fd514b7f8d433692021cbb63de9846 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_590c089b974e42b6a67fbd98067f9fc9 = L.marker(
                [40.92101669311523,-73.86511993408203],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ce157799ffc24aa4a638ee39fd5b579a = L.marker(
                [40.69133377075195,-74.1771011352539],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_aea41568d15f4bc49c0e87ba03dcbd7a = L.marker(
                [40.93108367919922,-73.8492202758789],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ebb5276fc721473fa4aa064e97bb2c62 = L.marker(
                [40.69510269165039,-74.1771240234375],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_3683827b7ca9440eb13c113fcfdfc59f = L.marker(
                [40.68777847290039,-74.18255615234375],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_2f2d698cf61544718a6efbb0ad279c36 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f66530e6a59b4754bda65b0c17f58f62 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_d29957a9d9884cee91a160ca338a185e = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_4c721ba9bf624b8ea485bc1aae9922ff = L.marker(
                [40.693748474121094,-74.1766128540039],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_08fe45ba7ec143d980dace4fc952ff90 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_0d7db8be17554c0485e88bfe4353c6ec = L.marker(
                [40.69465637207031,-74.17693328857422],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_89614d495a7e49119266154a52a0253c = L.marker(
                [40.65788650512695,-73.65257263183594],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_646690a9d2f7488e865f842493d2d2d9 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_655a5ec51e2f44b289db15a00b791141 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ba4a832a395844ac83b62503f059cdd3 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f115052b2d334e269eafd4c12965dfb2 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_9a8bc67df8024bd990aa8ef8f3e71b0e = L.marker(
                [40.694766998291016,-74.17697143554686],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_2bdf06b5b4744bb5a4c4327909641707 = L.marker(
                [40.72420120239258,-74.35098266601562],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_bc0c5d011c104653a04134e91bc9712a = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_9154bd6b61d841858e99dec2f8d3e2c3 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_de7b6144ee5b47e0ba12cbfca81fb9d1 = L.marker(
                [40.68771743774415,-74.18231201171875],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_2ec4046e42504c829465d71f9c55a4ca = L.marker(
                [40.69469833374024,-74.17687225341797],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_cdb2419268c643028e80a084fd102eba = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_aabc9a2813364251a0b843664b619e17 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f76ebf2eba0c474ab905731e33221acf = L.marker(
                [40.77683639526367,-73.5674819946289],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_9949a9c6ecb6454b9e81679b0459a759 = L.marker(
                [40.57231140136719,-74.12094116210938],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_4a76bf0996b842a5bb6e5a81f0030ff1 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f1731044d63b49699d48704ceb6f0274 = L.marker(
                [40.98601150512695,-73.76573944091797],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_570b56bab9a6417a8ef1f0b0bf9946d7 = L.marker(
                [40.63022232055664,-73.64573669433594],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_bd1457119afc40a48c568d04e72a6a08 = L.marker(
                [40.689266204833984,-74.18424987792969],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_59a3124e28ae43fbbadac1b9c14fc751 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_57bf8fa6f9b648c9bf51d273f69bae26 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_2566b47ddfee423780794eab9d30a225 = L.marker(
                [40.69509887695313,-74.17721557617188],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_52b35dfe0f794f559569a5bc459241d6 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_996c0d5bbc3f439aaf08ceaedca18e9c = L.marker(
                [41.069313049316406,-73.548828125],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_2e9fb2a1a2f74678832ddaa844216985 = L.marker(
                [40.69046020507813,-74.17755889892578],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_6c04bea766ce4c019c3f95cd02eb4e5b = L.marker(
                [40.69062042236328,-74.17750549316406],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_d049f06e6e984d549bab8456800b4ff8 = L.marker(
                [40.57637786865234,-73.95716094970702],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ab5b30c6ce3b45d4bde49634633a6b09 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1e20044056e7465994df1a185ef873ed = L.marker(
                [40.76484298706055,-73.53263092041014],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_fd6a940fa05240419baa67bffe06d432 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_822f47f8433242a392cccdf888d6b57c = L.marker(
                [40.6954231262207,-74.1775665283203],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_657a31e39a4f46e9942f184025bce1e8 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c18d93a3ab2c40b0bce4ec0d2bb2e343 = L.marker(
                [40.688232421875,-74.18350219726562],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_128451a8c27e463a90142bd8ffa6ad65 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_86dce2ace3a64555b9a6d70542e01575 = L.marker(
                [40.34009552001953,-74.30425262451173],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_d3806c8c290944bba3402359c3d19750 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f631e3bf3aeb4561aed55de5cd245fbc = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_684b27d8cdb84ff4b3652d33465be536 = L.marker(
                [40.87303161621094,-74.42591094970702],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_9c36c98c296b4e8fadffa5c25268cd51 = L.marker(
                [40.98587036132813,-73.80880737304686],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_eaf2017be70b48538212be45c71cfc86 = L.marker(
                [40.690101623535156,-74.17801666259766],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_36fb9c591ab544ac9396db0ad49ba3d1 = L.marker(
                [40.707195281982415,-73.5213623046875],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_204eaf15f6044030b63dade26e6bbd84 = L.marker(
                [40.690765380859375,-74.17742156982422],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_631538ad53304663bb9f9fd440a10aa7 = L.marker(
                [40.69314193725585,-74.1863021850586],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c0c6f9c574c94023bef9e62ae2562ed3 = L.marker(
                [40.69042205810547,-74.17765808105469],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_9304cee953d047038e9ab406b6089d2d = L.marker(
                [40.666179656982415,-73.62700653076173],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_6f6ef7c577cc44d29bb6fc9a86b2f56d = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f9c9705bbdef4bd9a8aa95cab2c9160c = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1a5f446561474b209695e2efccc03ee6 = L.marker(
                [40.6905403137207,-74.17749786376955],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_3ff36bfe4fa54892b62a72c07c5b7b15 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e1447b9e7b804d9e8646a104e7a8f6d0 = L.marker(
                [40.6878318786621,-74.18282318115234],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_846771360e9841d4914a11c33ec4b2db = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_918ee1d283ea4ea08c98bcf67364a5e1 = L.marker(
                [40.68812561035156,-74.18075561523438],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_4128aa59df424b78a728b7fa7db1ebf8 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_edf3dc8bc69f4aa68b7a24bfba9489c2 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_da35d73cd93c4f258bb8ce9b308e3d19 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_eff9cf68ca6d430f85fa8b8470d58b44 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1a3024f255b04dfd9a2455fdab54e37c = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_59b7360bf1b7426bbd2260ae39a9765f = L.marker(
                [40.59216690063477,-73.6338882446289],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_722f77e0ea904fd0ac521666080504bb = L.marker(
                [40.57548904418945,-73.98211669921875],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_19a748898a5c41cf8192a9003ff14985 = L.marker(
                [40.68770217895508,-74.18179321289062],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b7fcfdc7b7ed45d481ea149d52a395d0 = L.marker(
                [40.73226165771485,-74.16181945800781],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_fadca1deaf144a7797d09ba50cfb59e8 = L.marker(
                [40.68778991699219,-74.18206024169923],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ef0e699ca5744ca5a74883788000943c = L.marker(
                [40.72761535644531,-73.58158111572266],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_3545330dcc6e4582a52a5a3fced4b826 = L.marker(
                [40.935829162597656,-73.90230560302734],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_707647d2f8064b7c89e44f81b4c01c03 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_cd019a2d0ee946768e7895bd2309a0b2 = L.marker(
                [40.695289611816406,-74.17742156982422],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a9484ae5113e4954b531852512b7a0cd = L.marker(
                [40.6949577331543,-74.1771240234375],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_84e1d70c930742e2b5378c3b87bbcef5 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_73d500a7598b4dd3a6acb7e334c64c91 = L.marker(
                [40.69048690795898,-74.1778564453125],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_6bebb9a2c34743868586d35679546e37 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_4e6fea85ebec426e8b7132c7fb1bf47a = L.marker(
                [40.68769454956055,-74.18231964111328],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8b022a6becd24eddaec95f14851db3a2 = L.marker(
                [40.69525909423828,-74.17733001708984],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_af44c9e7ba57435e893c94f61cfc6497 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_fa1e083f903d44fa8f00a1bac28c2533 = L.marker(
                [40.68770980834961,-74.18148803710938],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e5ecf37c206245f0ac48358fb3dc0cc5 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_58dd8ce30ec44606b3477c99575f9e74 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_de97153810d046caaa584d8eadbf6cf7 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_fe1262ace74341208e18caf69de6eb79 = L.marker(
                [40.72880172729492,-73.4225082397461],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_36b42aa1584d4687b1090535af17d30c = L.marker(
                [40.920082092285156,-73.7686996459961],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_43648c209d794895bea499359f14655f = L.marker(
                [40.689807891845696,-74.17831420898438],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f0aa069cb9894436b9942846a437ef32 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_cece040f746c41cfaa25efa3b3530076 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b59beb29afbf4b78bd4b5e487d2d5314 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_67fca948d85343a39c455e40afdd4efd = L.marker(
                [40.695091247558594,-74.17724609375],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_d8a77b674405495f815c8b92492ceb63 = L.marker(
                [40.69314193725585,-74.1863021850586],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_64465448984c400dbca244a4e42d970f = L.marker(
                [40.69527053833008,-74.17739868164062],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_cf09a43724414a19976d1615da9d58b1 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e90d646c8da044fb9f2ef8cce37a66e3 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_82d5ac1ff0e540b1870086f540710c65 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_2e7961957af348b0a1f534c2930482c9 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b93a5beb6bff492c80385c62783675d5 = L.marker(
                [40.73617935180664,-73.4463882446289],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_5d7e42f9b7614a0f8159e236b97ec497 = L.marker(
                [40.56641006469727,-74.10482025146483],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a83fe257d7ad46929369e15da3145c6c = L.marker(
                [40.67791366577149,-73.61115264892578],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b4b0cdaaceb243b1a99da45b24424fd4 = L.marker(
                [41.033851623535156,-73.56607055664062],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_5110d907cbea48d48d426395e679d8d9 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_bbfe630859104742b4c75d3053a8701e = L.marker(
                [40.68837356567383,-74.1836395263672],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8060096d02bf4c0693755403ce0878f8 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_359da48f670a4418a3ce1afe8f80fb88 = L.marker(
                [40.69314193725585,-74.1863021850586],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_015cd41a4e134a768c0da989f507959a = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_fbe89d6c5fb1407b9408f59b9661936a = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_2298403234854ebd916a462872ac2364 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a64014c5dda9442cb1651bfe8c9bc546 = L.marker(
                [40.69050979614258,-74.17755889892578],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1171ebec7cbe4c03bf3a080f3d30d1fe = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_3db541470c9d4ac4bcb09939c20fcc93 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_128a3163e6d44ab2a29384cf459b147d = L.marker(
                [40.56075286865234,-73.91549682617188],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_eac578e51c524b41a8192bd987e1fedb = L.marker(
                [40.68320083618164,-73.68372344970702],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b2140f0ff3e546a0bdaa07e147ee067a = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_070a1de082b94c8fb06699648e131d0f = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f787b528b23c483cb391c67344485af8 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_63351018ef7b41498d9ddb631e588f1c = L.marker(
                [40.687984466552734,-74.18315124511719],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_550a6a40997247df94fd565042938344 = L.marker(
                [40.6909065246582,-74.17729949951173],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_dfcb757ec1264011a22e76ceb4d502d1 = L.marker(
                [40.701637268066406,-74.18387603759766],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a003e5d786154d8896efa398b194184c = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_07078d6f57dc4a3182c66eb2cfab4b50 = L.marker(
                [40.69520568847656,-74.17736053466798],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e4b7f63aeab945889b39bfbc91abd220 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8e788f52120b40ab8cd4db8463fb84e3 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1697f5babf5a4a9d92ba02f48c39204c = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_cc6476f3132c4ea1b5bcf7c96de2a83a = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_27ae8f8c498d46b7bdf699088df08e50 = L.marker(
                [40.93494033813477,-73.87889099121094],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_0df05a59ae7e4606a6c57ba1729a824a = L.marker(
                [40.76359939575195,-74.30373382568358],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_87a11c8f04894acf9078f1fe7d9e4cb8 = L.marker(
                [40.695499420166016,-74.17861938476562],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_fe42b8c116a143779a75649f7fa6cf73 = L.marker(
                [40.57580947875977,-73.95514678955078],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_025383beaadb41ddb30e24b4ebb9b686 = L.marker(
                [40.756690979003906,-73.64395904541014],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e1d18cf90024404ebef70adcfa0a50a3 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_2d076d6c86bc41dd9707c7317a2d0a66 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_063068f6a19f4aeead1955a33de147d4 = L.marker(
                [40.96425247192383,-73.68070220947266],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b7e1b85b325c48199130919e9e9a1f1b = L.marker(
                [40.9951286315918,-73.76197052001955],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_01a6ec05cce542bc95ee0490e5355c96 = L.marker(
                [40.73949813842773,-74.16783142089844],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_95fe25b4fce640b3a4b18edb9ea6621f = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_79a634562f1c4ab9aedf867d864f1568 = L.marker(
                [40.69497680664063,-74.17713928222656],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_0f423ec8268d4264af43987412709422 = L.marker(
                [40.69479370117188,-74.17704010009766],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_0d8ff69131f840179a58b5d6efc12c76 = L.marker(
                [40.57695007324219,-73.96514129638672],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ef579776a5854ca3b3053f3d69c1b605 = L.marker(
                [40.43175888061523,-74.24413299560547],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_57f18e1861b743f191de3012df84e005 = L.marker(
                [40.7608528137207,-74.2263946533203],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_7843d69ba6de426e81f8c55802604da8 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f1aa92d4cdb0407a98951b7e0c17d349 = L.marker(
                [40.76487731933594,-73.5324935913086],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c83c8d87094446ee9a6410afe9f7a0b8 = L.marker(
                [40.68767929077149,-74.18226623535155],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_7341af0e2eea491182361fce3eb2f9e5 = L.marker(
                [40.69508743286133,-74.17718505859375],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_35637affbee2443a92f2e3c424e1a5bd = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_3928df4a096440529f363ff58a6772a5 = L.marker(
                [40.695350646972656,-74.18035125732422],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_11142c676eb2412dabf4447a07449e4b = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_fdaf87db902b4f319a1e1b5f05ba26bb = L.marker(
                [40.6902732849121,-74.17790222167969],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_4bcc95c4a926425eb1cf7e70c12c4c57 = L.marker(
                [40.687721252441406,-74.18151092529298],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_91e04b26c414469c81cab4bfb8187f90 = L.marker(
                [40.7598991394043,-74.15282440185547],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_7f9863645bdd4dbb89d5ce06bff5b7cc = L.marker(
                [40.687984466552734,-74.18317413330078],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_7cc19fad81e84699b7a5d446c2fcf5b1 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_6a09e1e76cc64614b8f785a43f40b205 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_fc0e644bf2d04e199a892987a888208d = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_5316085cec3d42dca700c5646b8ec112 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_11ad613df7bf4d78bddafca28ea0319b = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_9167defe2a4c44868350d7c1a55fe37b = L.marker(
                [40.6951904296875,-74.17735290527344],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_033a2e1815254bf190782096920bef4d = L.marker(
                [40.92659378051758,-73.83845520019531],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_5a91745c227c44ae9139c6bc07e91858 = L.marker(
                [40.69459533691406,-74.17686462402342],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_562c7360ab8843038ff3689998fd8000 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_2c9b3d7dd5de4510a84ff1ae01809720 = L.marker(
                [40.92778015136719,-73.86337280273438],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8f77979ce170485085677a7dcc9b61c5 = L.marker(
                [40.93550109863281,-73.90213775634766],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_cb9a1aac59db4c85bf72ed80cba94d4e = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c67eac734fab44589a7fd8d2b714fd9f = L.marker(
                [40.76940536499024,-74.28118133544923],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f5d51fc8f7c741d3ae9a8ec2e135ba87 = L.marker(
                [40.77006149291992,-73.68224334716798],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_2b79d5b6d2cf4fce85edb525dc556d19 = L.marker(
                [40.69314193725585,-74.1863021850586],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_388f28e918834e5fa2f2171ab5a11700 = L.marker(
                [40.69499969482422,-74.18119812011719],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_94d6446ecc7f404f96f6b976c60ce011 = L.marker(
                [40.687721252441406,-74.18158721923827],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_7f4686d84fec4558a53f158d60e6ec40 = L.marker(
                [40.688743591308594,-74.18399047851562],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e17ce642ef304055899e7fe54555985e = L.marker(
                [40.69582748413085,-73.69305419921875],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_4409d6790606449db65e969be555d56e = L.marker(
                [40.79830551147461,-74.15188598632812],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8888f65463c24e3bbd9dcefaf74935b1 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_bbd1c37921d444889d548a5d59b61e22 = L.marker(
                [40.69314193725585,-74.1863021850586],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8c7a1d2c5a8746e78c5d9ad9111ef0a7 = L.marker(
                [40.85806655883789,-74.18350219726562],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_9208798f15264357803bb92ea771210f = L.marker(
                [40.720176696777344,-73.65096282958984],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f8e6c91f7ad44a0fb4bfa9ed464fe385 = L.marker(
                [41.50449752807617,-74.99958801269531],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_54b113f7fe5c4b46afcee8827b07f6be = L.marker(
                [40.694480895996094,-74.1769790649414],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_78a27ecc3559420e83179edc790733fc = L.marker(
                [40.619972229003906,-74.16024017333984],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a4ff8c7f56014ff5b69c6b6f19780780 = L.marker(
                [40.690242767333984,-74.17791748046875],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f71a9d6a85a14efb9432c6c99669d77f = L.marker(
                [40.693660736083984,-74.17682647705078],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_dffcbd4570a1469d8a51081542315f40 = L.marker(
                [40.687896728515625,-74.18285369873048],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b35e0dd28ead42c2b4e28da8340a5494 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_115c4bb8bad04299860fbb8f07e073a6 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ec4cfb48394d4f2c8c7877d5ffd2d697 = L.marker(
                [44.45609664916992,-70.51576232910156],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_23f21d8701354cc6b9c7b086732a36f3 = L.marker(
                [40.71134567260742,-74.17399597167969],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_65ff26fad015474eb4d7061c1ae8613f = L.marker(
                [40.69483184814453,-74.17705535888672],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_d0a791879c6c47f8b2ef8c8cf69200e5 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_816843ebcdf940268f15d1e99f33283a = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_fba5b0f2ca9f414a97c821d859b7474e = L.marker(
                [40.69506454467773,-74.17710876464844],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_5f9bd233e99848168cfa947f87592bd7 = L.marker(
                [40.69511795043945,-74.17713165283203],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_af87294e8c9b40d88b3a9da71e73c0ae = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_af19d2c466014a5fa4b01fd687c419ec = L.marker(
                [40.68989944458008,-74.17820739746094],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_6b0ed159540a427590c106816eb49e2c = L.marker(
                [40.5665397644043,-73.8920669555664],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_0bed9f651f254262813f65bdc9872d62 = L.marker(
                [40.69496154785156,-74.17708587646483],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b3a061f0d89e433293273168a3792928 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_d1e3013e93c54105b64e2121ec1f7cd9 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_07c94f6407144bddb008b0f77f76711e = L.marker(
                [40.71031188964844,-74.16277313232422],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ebd3eca13a9a4c0b84487888780aed17 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_dfcce4b71a5c47b5922501bec2af6e28 = L.marker(
                [40.68989562988281,-74.177978515625],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_057829d393e84723a884c99f354f8132 = L.marker(
                [40.68767166137695,-74.18144989013672],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a3e5247ab4284f199e290e24ef3c9b75 = L.marker(
                [40.69391632080078,-74.17668151855469],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_9803994068754b7e90e310a652d21537 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1fbbe0d36cb84ceb9b5dbf4b0b5b64b0 = L.marker(
                [40.69013595581055,-74.1778564453125],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_76dbfd73864d4b2d999603aab3bfb975 = L.marker(
                [40.69057083129883,-74.17750549316406],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_73edb393a175454bae9860c39d0f17b0 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_30494ad718b94125b4c92aaa4c4ce897 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_81eea32aff13421eb3c70df3650d8402 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_7a6a499ad6234f7f9dcbd176bc825724 = L.marker(
                [40.69214630126953,-74.18122863769531],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_24eadea3a8fb4826a15e1a95e96761ac = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_4106cf623b5f48caa8bac2325ea3f0ab = L.marker(
                [40.69441604614258,-74.17697143554686],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c89c8cfc896046ecba98da764583f6bd = L.marker(
                [40.693458557128906,-74.17665100097656],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_7da4b4065968478ebddc3ae3271f3256 = L.marker(
                [40.776451110839844,-73.46607971191406],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_6880d8f2cf67424f8117eae002117351 = L.marker(
                [40.68794631958008,-74.18303680419923],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_65ad18d7760144979f3239ac43146823 = L.marker(
                [40.69521713256836,-74.1773452758789],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_5bc8b5c4d3e948bda54559b381407af8 = L.marker(
                [40.69499969482422,-74.17713165283203],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_eb9d1d6e929a4b8b862f96d27e049f0d = L.marker(
                [40.6878318786621,-74.18269348144531],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_9081b0b45db94e4aa1896bdc86120457 = L.marker(
                [40.69529342651367,-74.17757415771484],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a2334bd0051f4fa4a090ee415e9d82a3 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_18550b54512a441fac480626fb6aa7d3 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_4f08a849dfc244ae84e95916d587051b = L.marker(
                [40.57575607299805,-73.96833038330078],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1ca5dbb995d64ebebfc984ac20cc771f = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_d166a8ffdb96403caeaf197e6e5c8bd9 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_dd354e458b6648788d7c852838d55557 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_564203e4fc6c455f9949b174220a6ba3 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c45d8127e99546f3be8394b0a91d8902 = L.marker(
                [40.69550704956055,-74.17786407470702],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8be66e4ff7424cf7be1e653546c28e03 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_7f3290a4f6424e05ac7d10a3a99e6828 = L.marker(
                [40.724098205566406,-73.65563201904298],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_deb57d9f664b4203954cd0df668f0a77 = L.marker(
                [40.69529342651367,-74.1773452758789],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_0ce4d1c03de948369a554939c0b2162f = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f2c90e15def641d988eb9d2813c076ed = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_6535b3293ae046308bd356d4f20bcade = L.marker(
                [40.57468795776367,-73.85514068603516],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_04eba80bdc8f43af89deba462fb2cd78 = L.marker(
                [40.69430160522461,-74.1768798828125],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_001e38f032d34228a0fc307c3f4cdb8a = L.marker(
                [40.68938827514648,-74.18426513671875],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_35ae36e24fb94fd29c66b2dba9692b41 = L.marker(
                [40.73244857788085,-73.67697143554686],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_deeeee35d293449f93bd70161b457f25 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_bf3d8fa65dbb429492aa34e9d3f96225 = L.marker(
                [40.68996047973633,-74.1781005859375],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_839a25885e6a4d10bfcb21f8cfe378bc = L.marker(
                [40.68843078613281,-74.1837158203125],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ff3f576d304d41fd86ea7baaf4e4bbb3 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_256b5610151d4bfc8fe0cb4849e3f8ff = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_6dcc4310a5464214901e843776527a1c = L.marker(
                [40.69499969482422,-74.17728424072266],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_374fc418114b4c3498325f64b1fa1bec = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b265f5d5a684400f8b8f31e5191ea8ff = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_933594bb0a084c159746b4bc73712938 = L.marker(
                [40.71751022338867,-73.61103820800781],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_bd1bc95504d04ea39d4a61f74b0b805b = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_cd5568ed78da4ed2ac9a7fc3f48cfd8b = L.marker(
                [40.71631240844727,-73.6032943725586],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_7ce4e7016dc24bcda9b287e90dffe391 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_5106a7a48e0b49d78c788960248e74d9 = L.marker(
                [41.02372360229492,-73.75814056396483],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_cafcbc7913a049888e3bd65a617e79de = L.marker(
                [40.57563400268555,-73.98138427734375],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_0c415094b4e9421a803ca064ae30649a = L.marker(
                [40.690528869628906,-74.17754364013672],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_da5f75f366734847815428951ccb94e6 = L.marker(
                [40.6949577331543,-74.17727661132812],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_cfc54f9d437b4dca89e8023fcc6cf4f2 = L.marker(
                [40.57910919189453,-74.15861511230467],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_0f7047a1b97645cbba66491fb894c980 = L.marker(
                [40.98594665527344,-73.80878448486328],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_d443de3bc4334eb88ff7abbe0ef3282b = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1dced11701f14b5aa9cdbb8f6c3b287e = L.marker(
                [40.57724380493164,-73.85749053955078],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_2e6e1b35fbd04936aa691ded8b94f3f8 = L.marker(
                [41.0347785949707,-73.58651733398438],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_6eeab6bfd57b4adebaa1ba30bc1db43c = L.marker(
                [40.68820571899415,-74.18339538574219],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ce28f8217fb64785b769702cd3092eb8 = L.marker(
                [40.70370864868164,-74.1841812133789],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a463bd18dae645f2aa385c5904c1aa0b = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b4ab8802bbcb4783a8c70e2ce8b551d3 = L.marker(
                [40.52697372436523,-74.1661605834961],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_41bc4c563873449998667116ced6aa57 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_78e2defe9bb74de4ae43a6dac99e6bc8 = L.marker(
                [40.68778991699219,-73.63213348388672],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_bdcb692782fc4b6cb55c1bf157313350 = L.marker(
                [40.692039489746094,-74.18130493164062],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8f0fb0c0bb854628a35bb4332b1c7a18 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_d5d3733e845e44f1bd84408cf56a305d = L.marker(
                [40.84197998046875,-74.19947052001955],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_806b7f71e27240d6af0f100192e3ea3b = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1ffd758aa25a461898e56aceee26a89f = L.marker(
                [40.98258972167969,-74.0533447265625],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_961d7dd0c6934d74a73bfd1bda7b488f = L.marker(
                [40.93640518188477,-73.84513092041014],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_0f136b6c350c443a95668a48b1120ba6 = L.marker(
                [40.69091033935547,-74.17736053466798],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ccf4aa362e4a4276aa884558ce35b45d = L.marker(
                [40.890621185302734,-74.26815032958984],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_aa86b2dcaf5e415e818faa3da0d559c3 = L.marker(
                [40.07609558105469,-74.74689483642578],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_bb4165fa69cd4597a03b966b3642a59f = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_fc1fbd0f1c484128bdcba32ae37babf6 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_2df8f31f6d4d487d85da7d14a3ef6fbe = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_d4e4e580037344128948f641f3ce72d8 = L.marker(
                [40.68772888183594,-74.18131256103516],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1dc47a9febdb488497814998e4d7bd04 = L.marker(
                [40.689918518066406,-74.18428802490234],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ebf614a52a2d47baa8243813dd73b177 = L.marker(
                [40.75444412231445,-73.68557739257811],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1cb406ce68b24a9ca115af79bdfe93f9 = L.marker(
                [40.919090270996094,-73.86743927001955],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_82e3f8603e74462982cfb345c1418ff2 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_7762764d791a40c599c0f8ffdbe06dc0 = L.marker(
                [40.69478225708008,-74.17694091796875],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8a6a3a971ddf431da13a2cd5fa174564 = L.marker(
                [40.68767547607422,-74.1817626953125],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_9085d577c9a14d3aaac699ada7963a53 = L.marker(
                [40.69095993041992,-74.1773910522461],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_3a10b45e29d2422f849b7e02ae21b82f = L.marker(
                [40.687782287597656,-74.1825942993164],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_9a90e68b7e4d41b8b1cae8711124c1b1 = L.marker(
                [40.69489288330078,-74.17708587646483],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a4e7a5347a794f0f83b925f9625dd6bf = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_7935ea22ae6244bc897d92345368fbfa = L.marker(
                [40.69523620605469,-74.17747497558594],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c8387a42117e46fbb24a4433569f1c60 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_5c9a74c9cd914e3c974d7ec62d74251c = L.marker(
                [40.688026428222656,-74.18321990966798],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_65a08d5dc9574d72850f222bc035fa7e = L.marker(
                [40.57186508178711,-73.86360931396483],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_d9de4729b01642448961e9fa9d7d1e80 = L.marker(
                [40.690521240234375,-74.17750549316406],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_59326c1374e64d44b97d145f34f55c67 = L.marker(
                [40.69525146484375,-74.17746734619139],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f1e93945941e456ab0d02041eb4d331e = L.marker(
                [40.69187927246094,-74.1838607788086],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_cb3d17a0661a44a89b37572754d49251 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_acedb00ae014493a857dfc72b9190dd5 = L.marker(
                [40.69548416137695,-74.17767333984375],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c79e8f6fa843493b91f18aabb7df083c = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_199e34daed124e9bb563d6e0d7f259fa = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8dd1757c351149c9b572970a0aea3c30 = L.marker(
                [40.68771362304688,-74.18192291259766],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_d2e056b65092484a9420542c88678930 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_69d4c97b8c3f4f7bb264987dc2eeba04 = L.marker(
                [40.7031478881836,-74.18366241455078],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_9d4a8606a47d43cba849deb0be2010a2 = L.marker(
                [40.669700622558594,-73.55374908447266],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_3e9d08d88a3d4409b260dea4a1715d9c = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_039cda64ed9d47a181e3e7abe48827ac = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c6c6ee33777747348a30207da1400a9a = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_d938c3dc1fae4561a01c3f2291a58e70 = L.marker(
                [40.7648811340332,-73.53257751464845],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8bbbdfb7cf9043089c699bc320e8b2c1 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_526899d3772a41e9a840bfd3384f5df1 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_955779de359142989770039807ffd159 = L.marker(
                [40.61316299438477,-74.44533538818358],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a64ae897559e4d61b87ef2f72cac8a05 = L.marker(
                [40.57609939575195,-73.84117126464845],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_d547e33c9c7d4e56a38448a41c0098ad = L.marker(
                [40.69523620605469,-74.1773910522461],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_0dee96d8d68a442fb1e272b0132711e8 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_fd62db6dd74844ecb04be386c41893ad = L.marker(
                [40.69314193725585,-74.1863021850586],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e79bd153008f459484d656d2697f970b = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_d852fd369df64851929cbd373b3fac26 = L.marker(
                [40.69438934326172,-74.17678833007811],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8742f8c3abc149acb7ebb0de95ed6c49 = L.marker(
                [40.694820404052734,-74.17724609375],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_6b7ac4c7dce24ec39169ff73e4995891 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_73592636bf3140a680277985e505d1e8 = L.marker(
                [40.69062805175781,-74.17748260498048],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_41c8a01b858845ed90ccd25741a0b8bb = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_12a9106c574c4edd8278df1fa5da0d4c = L.marker(
                [40.69511032104492,-74.17721557617188],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c17fd7975ebb41dd9742662b134b55bd = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f077c827f5b746ac9bb42270f701bf94 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_bf5430c8408f4b0a8d8b3e2204156bb9 = L.marker(
                [41.01092147827149,-73.91253662109375],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_3170015a81994a9fa516984751631128 = L.marker(
                [40.694618225097656,-74.17692565917969],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b13debd6cc2643808943a59d78c7bb54 = L.marker(
                [40.69514846801758,-74.1772232055664],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_d520547d5947410d9e0258316446b270 = L.marker(
                [41.01962661743164,-73.6243667602539],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_728f8d4cdda7425d8f156e5de82ca6dc = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_acda66b427a24ba491903fcabaf596df = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_4ece751ca8a84bfd8193532ee6ccae15 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c05d9ac16c8443b2b6a18099f0537628 = L.marker(
                [40.6933708190918,-74.18842315673827],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_009ec9f52fb547e587f145cbeef38df2 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_afb2e322471242e5808d2a1cb0624e1f = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e2fc07d1f3d44246a62b2c45c076983b = L.marker(
                [40.69551467895508,-74.17777252197266],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_16bf0be2f49c4e279575a3b622264a5f = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_feed12c2e8b748bb84115cf6d0340f21 = L.marker(
                [40.69075393676758,-74.17735290527344],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_979759ad3a0048b3ad3f230e6087fc75 = L.marker(
                [40.9499626159668,-73.84806060791014],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ab5656dbbb4e4c39a7eb068116852b0d = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_af9023f7b2b34101997f9894ddd30b27 = L.marker(
                [40.69314193725585,-74.1863021850586],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_d3fdc3a68837413c9e8d6d667252674d = L.marker(
                [40.69378662109375,-74.17667388916014],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_201f4b6ac716433582c6564bb3620982 = L.marker(
                [40.91913986206055,-73.88695526123048],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1560f0ed18e94755b3bc71a47f835744 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_5a8602645f1c4b4e97bbab885ae4d047 = L.marker(
                [40.69410705566406,-74.17684936523438],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_dc1b492f8f6247aabfb21a6a1b66416d = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_717fadeff8e74d9fac4805181f318c8d = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ea0a768b9d3e4131a59a5779ed2095ca = L.marker(
                [41.010269165039055,-73.79814910888672],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8d6a3ae105ea4ab98b598a83de4106cc = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b6c6736bb7be4630adc21c8c074ad501 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_7c91a2980a804e629d21791103673b0f = L.marker(
                [40.830581665039055,-73.69227600097656],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_85d79c3a5cd149118187dfb5ba149592 = L.marker(
                [40.69369888305664,-74.17675018310547],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_054f4a64e61a4a318b04a31014c868c5 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_5e669d0e0f734c7596d0c98b7f33a198 = L.marker(
                [40.69477844238281,-74.17694091796875],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_936c98c11b3b404cbf4f4011cfafd208 = L.marker(
                [40.57573318481445,-73.96112060546875],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_6eee45f5125d45e9bb192370a8befbab = L.marker(
                [40.5764045715332,-73.95724487304686],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c1177c4f8f6045308cb6c5a70c4d15dc = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b664c88a874a4b14969894bb60627fcb = L.marker(
                [40.69485473632813,-74.17704010009766],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_3d0c93b5269340829f3b4704db01e123 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_2111a8346627420b85130874a084c115 = L.marker(
                [40.68817138671875,-74.18344116210938],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_aeee6a45cda342af9727bcb67d689c59 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_4980ed47a9ea4f489b5cbc627ad26b0f = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_d85186dd9ec54ad78909d62641214481 = L.marker(
                [40.693809509277344,-74.1767578125],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ed642c9339cc40f7a44d82e29a9933ca = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_85eec27d42884ed8818a4d1c0bfdfac7 = L.marker(
                [40.7239990234375,-73.58760070800781],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_eac217b5504e49709028d02c06e2e310 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_aa40220b2e2041e382984cc1a368a51d = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_9fe648798c6e49dc9336313ea019fc39 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_caf086b9a54a4f22bc22ca4e0c55ae59 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_38361128aa9d44bb9a8e86b05568e601 = L.marker(
                [40.69511795043945,-74.17728424072266],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_bda8f2ee215748f0b0c967a07a641c60 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_7f57aefa41c54e928d26fa9f39030f8d = L.marker(
                [40.68774032592773,-74.18180847167969],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ceaf6de00b5340eb8aef6bdb37a840b8 = L.marker(
                [40.68770599365234,-74.18142700195312],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c6c918d736a442eeba1fdc323096d1b4 = L.marker(
                [40.6945915222168,-74.17700958251955],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_0e52e249c8954f10bb6c2511941cee64 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_42ca6bef820a4e8783733f819f18890f = L.marker(
                [40.68778610229492,-74.18191528320312],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_db4274f5376e464099e502f35500b468 = L.marker(
                [40.86845016479492,-74.19135284423827],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_aa8afc4d9f5b479f955a4ab61e4993cc = L.marker(
                [40.6877555847168,-74.18158721923827],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8ecfac2095e74e499285243e005530b1 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_bb749b9d85214e47b3bcd88624799f54 = L.marker(
                [40.6888427734375,-74.17919921875],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ad4509e0e62b48e5b15aabb28653ed4d = L.marker(
                [40.69475555419922,-74.17696380615234],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c19fed7b52ac4090905c3204d0bc1218 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_119c44f38d3a4dcc99f6e422686e951f = L.marker(
                [41.04362869262695,-74.06350708007811],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ce8b18d70fad4f87a2afc81a0dfe3e87 = L.marker(
                [40.576416015625,-73.98625946044923],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_47d238a5029142a595b57b05d5d55f55 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f1a48ce96c4c4d73856034a98a13e400 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f54bcd2aaaad42629b9356e5745fc211 = L.marker(
                [40.76745986938477,-73.699951171875],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c34d6c5c40b4441e8e9fe57624175a0b = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c1ed4d64cf99476eb5bdfbd6ce8babdc = L.marker(
                [40.69021606445313,-74.17769622802734],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_7211580529284e79af525db81fd3a59c = L.marker(
                [40.69486618041992,-74.17705535888672],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_74add968e1474d4ba34ada6aa9c0323f = L.marker(
                [40.57709884643555,-73.9732666015625],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_65b7a7db2eca432ab8d5cb58c0c4bea8 = L.marker(
                [40.57716751098633,-73.96849060058595],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_438b5983a827409d98a175950b482b75 = L.marker(
                [40.69495010375977,-74.17720031738281],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_37305dbc142945698632400ab4cafe49 = L.marker(
                [40.69074630737305,-74.18421173095702],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_77835207fdf6423da5bb79d2efdbdb01 = L.marker(
                [40.695220947265625,-74.17726898193358],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c4cc86c4709e4aa9ad3ea648bec2b792 = L.marker(
                [40.72795867919922,-73.66849517822266],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_364c95a6c5364038aa293cfb9e8d8c9b = L.marker(
                [40.69032669067383,-74.17772674560547],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_d7794fc3f1de4fd1b3d3825504c4d851 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_245830ac7e404d0287b6ddca3227a5c1 = L.marker(
                [40.69494247436523,-74.17723083496094],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8fe1efbf9bc442048bd2ed486ec2048e = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_5846234a518e47908f23ccc295e5ef3c = L.marker(
                [40.691383361816406,-74.18397521972656],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1e55f1f703b444008c22f311cf0ec97e = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_03820fdbdb6c432aa26de52f6b1fd656 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ab8496e817cb4923827f3bc2e7c937c8 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c808304911e24908a2f205a12e45adfc = L.marker(
                [40.6949348449707,-74.17707061767578],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_394c86335f9b4e60810f8dd6d93bd97d = L.marker(
                [40.76900863647461,-73.65055847167969],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_29bef9e28940467cbc69a7b7deb7a52c = L.marker(
                [40.408016204833984,-73.99764251708984],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_6947a1068a034909bfb90342bca1b079 = L.marker(
                [40.68772888183594,-74.18232727050781],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_d7c68124ca2f46b78db3757efbad70de = L.marker(
                [40.6907958984375,-74.1773681640625],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_7c77d087acc94b998bdc1cfa9dd021aa = L.marker(
                [40.69138717651367,-74.1769790649414],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_915ea100553349d7863d781018f12802 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b2a5a1e3593f408baa6926cf607e0df2 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_cb48c56e37f34f15b925953204311993 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_4b9826e5ad564001a0c52f99526eb254 = L.marker(
                [40.76482772827149,-73.53260803222656],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_45ea00d3d28a4b76a4d7a128912cfb35 = L.marker(
                [40.69513320922852,-74.17729949951173],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b770ecde24ec430cbd7982932753de71 = L.marker(
                [40.69013977050781,-74.1777801513672],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1ded8dc3e2b440cd955b399d1ad35ba9 = L.marker(
                [40.7996482849121,-73.66400909423827],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_6ba2f43dd80c4d91a1801ef8c3b5a066 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e712149e652647f8b704b3d316e89ab8 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_dc93eeb7bfe74e959f58d0807d2b1621 = L.marker(
                [40.69015884399415,-74.17781066894531],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_458231d13c2d496b8d00b52dae8b58aa = L.marker(
                [40.69477462768555,-74.17691040039062],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1c68161c4aa045e3b260278c83f83f55 = L.marker(
                [40.9940185546875,-73.82162475585938],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_0f6a9a8fb5654181881f0a21714e26ba = L.marker(
                [40.95001220703125,-73.83389282226562],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e4188268eb874ca4ae87dcb8bdc64c80 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a8739019ab894f0e8057aaf867ee98f9 = L.marker(
                [40.69552993774415,-74.17764282226562],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_5c088ec727fe47a8b50bf45fc09053ee = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_68da85cc764e45a18b53279d9750b24b = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_2450b5d5833648b2bcf170370e273605 = L.marker(
                [40.69004821777344,-74.17792510986328],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_75cafa6b7d0f48e298b2a5e4a80eda27 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_76862b63dfda45f98781d9a16ff75870 = L.marker(
                [40.587589263916016,-73.67053985595702],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_aa5964291c8a4ea5b29689b420b4a553 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_2b03201c79bb402a8ad702f51c7739b2 = L.marker(
                [40.69546127319336,-74.17765045166014],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f44107dab77e4f77ba5cbb6cd31674d9 = L.marker(
                [40.80820083618164,-73.67659759521484],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_21215d15fa6547d686d82bb6dbd7f196 = L.marker(
                [40.695125579833984,-74.17725372314453],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_0ebc6e31ca7243d28e30410be6617e0b = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_13ecbc52746244e29cf0c0f9b75a9f1d = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_15647d723ca54cfb8929250d6ef2fb8a = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_35e58eba358b434f958658ed0c99887d = L.marker(
                [40.69527816772461,-74.17749786376955],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_5c1b16dbd3264b09abef9a00b1104eda = L.marker(
                [40.69514465332031,-74.17727661132812],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f3c1231c3aa64b55a023395caac6eed4 = L.marker(
                [40.8941535949707,-74.15949249267578],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_120fae80d0ba4c8c87abf2881e516d1a = L.marker(
                [40.74185180664063,-73.6087417602539],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_905c4da516974326aed51958690bbbca = L.marker(
                [40.695655822753906,-74.17844390869139],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1b0510caf59347558936961c4e17530c = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_46d10c9d74704b9a96468bbfd689a402 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8270cd85d58946cd9b6ecbd9e6697b02 = L.marker(
                [40.694499969482415,-74.17684936523438],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f6c2766c96ed4243ae726a1d72645b04 = L.marker(
                [40.69158172607422,-74.17721557617188],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_2007003f27f0475f93c1c627a723e6fa = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_103ef8e337a74c8990c7ff43772ea366 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e6d4775947b5418c8136c17261468dca = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_2a4df6e9fa484abeb14daef18e470dcf = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_6037cb5cc29b42138e74e680d1a112af = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_51a21b534cf7485a8e9b0190756c0e5a = L.marker(
                [40.942291259765625,-74.07099151611328],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_aced289d822f4d2d9da9b8d87af9e1e5 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_90594f4274bc439b83f614f8ea4d6ea4 = L.marker(
                [40.78792190551758,-73.5022964477539],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_d1117036dffd49999cd316a86bd57c72 = L.marker(
                [40.69496154785156,-74.17710876464844],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b94f291042bd4a2caee9350de6384b6a = L.marker(
                [40.78753280639648,-74.25613403320312],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_696d472962564542830b3abcc56cbac7 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a0e5c66902434ab39d7a824d77b0b0c6 = L.marker(
                [40.925880432128906,-73.84355163574219],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_594653dc9da540afb690e06c91f59d8b = L.marker(
                [40.64891815185547,-73.6732864379883],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_78f44b83d85a4137b0808eefd3a2b6b2 = L.marker(
                [40.69009399414063,-74.17778778076173],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_449edb47e439475dae8a103340e568df = L.marker(
                [40.69548416137695,-74.17798614501955],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b463c069f6a94690af6ae0b028abbb23 = L.marker(
                [40.69026565551758,-74.17781066894531],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e4ecb3b890c347c2be2c087028e25868 = L.marker(
                [40.695709228515625,-74.17813110351562],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a8d66e99d6e8421983494f67a56ad594 = L.marker(
                [40.71096801757813,-74.16207122802734],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_9d7db23917cd40ee9b44999d7874e259 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_874007df705d4b20b1d1e6302a49c316 = L.marker(
                [40.690528869628906,-74.17755889892578],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_efe483dace1a468cad611543052081dc = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_06a9e162c6694a7f8b8e04c7e7711f39 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_968c213453f94b9cad3bcef54e6109b6 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_4adb1df857ee45929779931fc62807ec = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_5f2f2337f9cd4d99a7f68570029d5cbf = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_6c709d7d690d404891581af4f1adc437 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_5ee5540e21214e09a2a10708c6f20836 = L.marker(
                [40.72895050048828,-74.1622314453125],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_be0bf2b1e119485483b2d9e3d4091e4a = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_40b2025b016d4def970fe83730885b88 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1689ab4d4f244b76b19ea42d78dbdcc3 = L.marker(
                [40.69503784179688,-74.17726135253906],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a3765472fb704a12a4827478ba7ef172 = L.marker(
                [40.69314193725585,-74.1863021850586],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_df82674c979d479f9bef4e9846865819 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_683fe049011b4999894d4677b0a9c355 = L.marker(
                [40.69445419311523,-74.17681121826173],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_520a0e63207d4b94967c4fcfc9bb241d = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_946f492d65264d28884dd079849b3f3d = L.marker(
                [41.01601028442383,-73.77580261230467],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_53a14c7b382141a09b98079eabc0740d = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_97bf8987eb08411c896e6d7f65b942a0 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_9062b2b94ef2491f999406364f784098 = L.marker(
                [40.69572830200195,-74.17909240722656],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1af823dee7ce456a839d5f47f0bbb669 = L.marker(
                [40.69459533691406,-74.17701721191406],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_3305ce376b7046eba2dab260b05ad267 = L.marker(
                [40.694984436035156,-74.17707061767578],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_14a44e3d056940faa2ea540765ee7902 = L.marker(
                [40.73969268798828,-74.24847412109375],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_7e1650bb7991462fad87921bca0f06b9 = L.marker(
                [40.74010848999024,-74.15618133544923],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_572e1d9ca69b40a68c7b8dcefd4339ac = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_aef822b2973e45e3b3109b4d65ccdc68 = L.marker(
                [40.69532012939453,-74.17738342285155],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_12c19a53cb934108986d50ef2984c2b6 = L.marker(
                [40.69480895996094,-74.17695617675781],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_201b711f266a4a0e9c2fd631b93b1bb2 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_4406f2381a37431796ae419221d8497e = L.marker(
                [40.693214416503906,-74.1766586303711],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_4f58ec02fa454cf1821e7b55acfae498 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_34d44c59e69640f58df1625a9dad78a4 = L.marker(
                [40.6828727722168,-73.6729965209961],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_34d5e0c212a64397a12578b3b3f1621c = L.marker(
                [41.3167839050293,-74.13088226318358],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_4dfd2901f07a415b897889372558bc88 = L.marker(
                [40.63020706176758,-74.15245819091797],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_6cc1358784a546d49406557e5d5e1b4d = L.marker(
                [40.793601989746094,-73.67040252685547],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_2f20b96ad445447cb83b048726250f75 = L.marker(
                [40.69118118286133,-74.17715454101562],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b5f7f7167ad1415dbeeb06d59533678d = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_5c1ea4e9f04a4dee9f4e8d340e5b0673 = L.marker(
                [40.95256042480469,-73.83452606201173],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c8a201251baa43df99105b8a0125e69b = L.marker(
                [40.57569122314453,-73.84269714355469],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f2104537efac4dc697164d90f7861d3c = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_6deec53254894c6aaecf59b660699105 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e512094513ed44dd878f76ad1c073a04 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_88708e7cc90248dea926a38eb4bffefb = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_fa4ef835299b4b27a32f4489dbfe213d = L.marker(
                [40.68801879882813,-74.18318939208984],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b027289569504200a05468b91be9b444 = L.marker(
                [40.695743560791016,-74.17813110351562],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_70a4397779404761a45dfce547c5bb89 = L.marker(
                [40.68788146972656,-74.18297576904298],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_801205b11a0841f6b9621c2204e7685f = L.marker(
                [40.68770217895508,-74.1819610595703],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_6e11a346417b4055b2888f67d6dce090 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_9b053c0a55e14cdfa9e8b7e7544cd44b = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a6a4244e7ac84239b2f69ee3928b4d63 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b52bb566753a44049b7708493ab105cd = L.marker(
                [40.68772888183594,-74.1815185546875],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_fe8a4c83cd3547cf88a7b7e590e988af = L.marker(
                [40.6949462890625,-74.17713165283203],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1a9020f7a9b840828e0d51b1be177069 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_29df644f87764ce8bd3661dcaf098e14 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ac551e4fa5d84caea1dcb41a1e62486a = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_5c55d31ab4584029856edf64094e7369 = L.marker(
                [40.69512176513672,-74.17726135253906],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8fb6357a98a348fe99f43f8dcdab4d28 = L.marker(
                [40.98974609375,-74.2349624633789],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_2cf3e72614304174a94c5378c737dc2b = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_4ff24041b8804138b9038c78c58b095a = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b11329143f3f4718b5c3919c0cf44144 = L.marker(
                [40.68808364868164,-74.18331909179686],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e6bc5a482ea44748b448d6d7daf825e2 = L.marker(
                [41.06896209716797,-73.84562683105469],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_eb34fd9518594440b8ed1d2df3dace2b = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e23537da29814270853efafad74c2ea8 = L.marker(
                [40.687744140625,-74.18146514892578],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_2f8ce14cf73c473899da4171e95e9101 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_4867ccde34ae4196ba1f1231abb2374b = L.marker(
                [40.75770568847656,-73.66199493408203],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_02c5bddd85984d7ca3a74b5be0902252 = L.marker(
                [40.68794631958008,-74.18255615234375],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_5921d41cc5894b8ca59d918633d47e19 = L.marker(
                [40.68880462646485,-74.17922973632812],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_34faf589434a4b57aeb02654022d4fc6 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_574b4f2856e74b6695d483331b2f0e41 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_3980e3be66f24809926058fe0184aade = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_4b8e4f5bcbee41a9a05014ed5a29c050 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_07ceab5e062547d2a0bb11586dc0785c = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_627d3647903041c7af23d5932e2c46df = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e9a0856e0b1a48fb9381698f76340eb6 = L.marker(
                [40.69500732421875,-74.17700958251955],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_4913f888a76a4f019de3b69d1284ec15 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a91314012c574122aefdacc3fba45285 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_4efb55793ff1443b92f9fb5d6b08c60a = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8e0455a96f054c418e6186f3f971c8a6 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e0da7ee1d8ca45e8952f34d50f228f6f = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c75ae36db24649b381bfb002fbff33af = L.marker(
                [40.69537734985352,-74.17770385742188],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_24e64c9202bc46f3a4993bb0242c87f2 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_47bb0f0dda8b42d893781c6bdeb411a6 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1b7db0eca77043f4a9d0bb02d6208095 = L.marker(
                [40.69010925292969,-74.17778778076173],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_5d59498fe59841eb90b25e8a2f1e65b4 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_aa7f28ffb25f4bb19a81208eaccfbde7 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_718d2c0c033b4f14b26b6feb1e69ef09 = L.marker(
                [40.57579040527344,-73.994384765625],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c2bf6d7791d740ca96c74042f8ffad7a = L.marker(
                [40.69009399414063,-74.17788696289062],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e81582f349e446439231a862ebef52fb = L.marker(
                [40.69314193725585,-74.1863021850586],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_44756ba6aa014abcb99fa4dd8333dca1 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f8478e8373ad4e06a764a47a8f443d3b = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_4c821a357cfc40ff886a5771a1a397b6 = L.marker(
                [40.69520950317383,-74.17726898193358],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_75780f0adaca4a8fbf68a47102bfbb97 = L.marker(
                [40.9925193786621,-73.97752380371094],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_7004d4a52a884f81888d46ff59a86efc = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_0e1cc593e99442768910c01ee7b58580 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b5da4439d77f461c8a3f0a2926232294 = L.marker(
                [40.6879997253418,-74.18324279785155],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_9a41ae96681a4cf7a0da6b3ad0698d2c = L.marker(
                [40.694278717041016,-74.17688751220702],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_3728299669cd43878b545749d326b5c4 = L.marker(
                [40.68777084350585,-74.18257904052734],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1f4d46000a3c4f32bcdada2e5041dea4 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e78b657cffcc47638fe9d95aebb061bf = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_62e3af0c186b4da6a52fe485f21e014b = L.marker(
                [40.722206115722656,-74.34844207763672],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_2092462b91eb489780fede324cb2f110 = L.marker(
                [40.7038459777832,-74.16366577148438],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c6dfff6dcc9844e1a8205a5d67fbf107 = L.marker(
                [40.68767929077149,-74.18170166015625],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_6b4f0a475f764118aa27ff31ab54e0a4 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a38e89e4d960468d94edf97e2d480804 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a446cd2f8427408c8eee26d1705d6ff5 = L.marker(
                [40.69529342651367,-74.1773910522461],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b0cdc6d648a042ebba92edd86f643344 = L.marker(
                [40.695106506347656,-74.17723846435547],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_963d116f854c4eb39f1296c20084008b = L.marker(
                [40.69031143188477,-74.17768096923827],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_11ecf1ee25834d6a9db1bdc6a85959b7 = L.marker(
                [40.69403839111328,-74.17684936523438],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_dd627a45f425421597c583be2444e7f3 = L.marker(
                [40.68768310546875,-74.18157196044923],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_34f5cba9793242c6976f90c5f737911e = L.marker(
                [40.695625305175774,-74.1789779663086],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_7a87e0777d5a4413b734c2f492916c6d = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_156993118105400c85738eaa38306d3b = L.marker(
                [40.69494247436523,-74.1770782470703],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_22c769e7830340418725a23f6cc97cf5 = L.marker(
                [40.69517135620117,-74.17729949951173],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_138d6b12c3bf49d2acf6de178e1edf30 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a098dd72f24645e88ea0c11c915d09a8 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_fc5e3b78880841549c382dd729b7c267 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_11ff235878d6495c876a46f27d806b80 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_bdeafc81d0884f1698b3399932300410 = L.marker(
                [40.69489288330078,-74.17703247070312],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_dfe19c53eb8141d9a2624f4cd2acec64 = L.marker(
                [40.689277648925774,-74.18579864501955],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b1592dc6c8cb452c883940c5e609d176 = L.marker(
                [42.1362190246582,-67.41030120849611],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e128967f95a04f8e8ef4e6405b2cbff0 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a7946e585f1b47378d1c40a251ac8d3d = L.marker(
                [40.53467178344727,-74.21947479248048],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_150f97360a8a4c819fc8a6f83dfa5980 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_dec1d6eee4da4316aafb3d93cdfe95ba = L.marker(
                [40.94395065307617,-73.87395477294923],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e77f3c5df8c743108f9d5043ce7abb58 = L.marker(
                [40.78903579711913,-74.51585388183594],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_0167b32aa47745a193f24e05bb7a45ad = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1d26aff1b34a48ea82fde061767e3337 = L.marker(
                [40.688026428222656,-74.18334197998048],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_4f15b39a8220428fbcfbeda45f46baeb = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_16d312c26ac748ec856bbf3aee50225a = L.marker(
                [40.69502639770508,-74.17710876464844],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ddf5a14908b944d4a7edb73ecd26c3c3 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_500d5cbde83046aab7a79d2e08cff95a = L.marker(
                [40.71765518188477,-73.66395568847656],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c8bee2b1d9284026be869f18140d0044 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1f4a35fa72cb4d2f9921856fc0718fc1 = L.marker(
                [40.69064331054688,-74.18425750732422],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_6f2edf37131541538535a334d7a6ed8c = L.marker(
                [40.69472122192383,-74.177001953125],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_845e699a1b5c4dee894b6d7c927362d0 = L.marker(
                [40.694969177246094,-74.17700958251955],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_4d8442e20d084be9ad0447c6e4a7b194 = L.marker(
                [40.68767547607422,-74.18182373046875],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_497f4f05a2d3451ebdd4789ac4b8c3a4 = L.marker(
                [40.63920974731445,-73.68767547607422],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_6d32d8a79f8a4b708bf2cf228927e6df = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c1f7fcbdd2ff45ff9d0ede62ba0cb215 = L.marker(
                [40.77569580078125,-73.5618896484375],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_2b6d9fddcdcc467589f87504c30bf515 = L.marker(
                [40.80055236816406,-73.68228912353516],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_18ae0699f99f4eb89cfa50c47e96233e = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_6a1560cc70424cd5b7fde8d7e299fb31 = L.marker(
                [40.693523406982415,-74.17666625976562],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_7e8cc2bf2b674deab767dae431ecbae5 = L.marker(
                [40.69523239135742,-74.17711639404298],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_9278275af9c544358069fbfa0a80f94b = L.marker(
                [40.69528198242188,-74.17764282226562],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_df128fac40984ebc96e0d2a78edf965c = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_42b7af38f5494b1c943bab21f8e586dd = L.marker(
                [40.69077682495117,-74.17738342285155],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f50b21d8abf6461e8535c72d26297e55 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_d23f83b85de54d2f996026cfe7f67bf6 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_5a757e3eddd7401198810ddf9660ea36 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_9e7405789e1c4f61ad11d1d05c126983 = L.marker(
                [40.69500732421875,-74.1770782470703],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_826ae1985dc1436ba96c3dd1480a6760 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c7c940d76cd14857891834e0d9816a36 = L.marker(
                [40.57587814331055,-73.9552230834961],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_7da7079bab79437083086568866ad26d = L.marker(
                [40.687950134277344,-74.1830825805664],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f552e8a8dbc84bc1bafb7a7a9460ecf6 = L.marker(
                [40.65538024902344,-73.66626739501955],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_290bba75120e4756ae7eb1258ed0704e = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_99326df16c1348aba98810daa91cade7 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_0123523374a84c82aa41f614896eee60 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_0ddf50e6d0ec4236afca0ac57afea30f = L.marker(
                [40.692115783691406,-74.18126678466798],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_fe0bfa0c8c8547c6a3ba669b06035c71 = L.marker(
                [41.00914764404297,-73.76945495605469],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_11f4755710fd436799e62490af2d4077 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ddb0f5d958e6422c95727d6c15470f41 = L.marker(
                [40.68986511230469,-74.17813873291014],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a1233e67539442799ca29b56c6437b0a = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_29cabdc4d0484c3d8e3b71712085a0a9 = L.marker(
                [40.645320892333984,-73.61393737792969],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_84410b95d15a4c63a96f080ebc423e42 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1ea390f62067484c9c8ed042ba2410ef = L.marker(
                [40.68778610229492,-74.18272399902342],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_9b0cb6a9abb3464487b08c91c9c17ece = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_128482c75711474bbdd9e3ffedeca09b = L.marker(
                [40.69084167480469,-74.17738342285155],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_cf1b0a64e31b4e65ac2b95ae11749a31 = L.marker(
                [41.05583572387695,-73.5430908203125],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_2415797bdc0e45eb9454254543098ce0 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_567f9a13db4c4afcaedbf6cc84c75bd5 = L.marker(
                [40.687801361083984,-74.1825180053711],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_31605890721e44a6b3fe03b1c330e313 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a7c55f4002454d4eb15fb6b78e86f3d4 = L.marker(
                [40.695213317871094,-74.17729187011719],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ae3c162cf30647baa4f4d0578bba54aa = L.marker(
                [41.2952537536621,-73.7833251953125],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_6c2e6b65e15b431db59b22d69534aa59 = L.marker(
                [40.68806076049805,-74.18321228027342],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_25ba6646580146cdac99cd14f6fb4f8f = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_74ac7987246e4c3b8027f96c1b6957cb = L.marker(
                [40.75199508666992,-73.54563140869139],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_d9482f5c62504f73b30f8996bdcc1eb0 = L.marker(
                [40.57405090332031,-73.8509063720703],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_669bd04bf3c041cf9525185ec46f545b = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_fc5c19e5fb6e4baabd35453019b81b6e = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_5719b5e30b224a308d725a955bae10b5 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f544217de3634c7181edadd2335f00a8 = L.marker(
                [40.69079208374024,-74.17733764648438],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_9a1f0831e49e4f89b2b7ccf3d475d01d = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c14e7e8c1c2b46fb9c205d90ccf9ea13 = L.marker(
                [40.74564743041992,-73.64237213134766],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_3cf9b4175ab94f05b9d46dee64204168 = L.marker(
                [40.69515228271485,-74.1772232055664],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e523e80e6269421cbdf4c632580c3ca6 = L.marker(
                [40.69011688232422,-74.17793273925781],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_62f63b1cdb5242939ea0f7c40adebbcc = L.marker(
                [40.695667266845696,-74.17802429199219],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_dabc97f4df5c47aa9eee08ffccfad0f7 = L.marker(
                [40.695499420166016,-74.17772674560547],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8bd138ccb72f47d891f0bc4142fbe2a8 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c0174ecb57ae48f192c90ba0d0a9d00f = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_83278a96d2c1482da9ca27db84eda68f = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a0bf989c902f43d7b3af824c55ec04cb = L.marker(
                [40.694522857666016,-74.17676544189453],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_eb048184dad64b609329f86409774a35 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ea322da16ef4462ebeb5f9270de7c8a5 = L.marker(
                [41.01694107055664,-73.71806335449219],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_7ef55469d5ac49f2abd466ee8a5c12d4 = L.marker(
                [40.796241760253906,-73.67312622070312],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_6e08bb8e89ef408c9cf847e0a9295bc0 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_cefd55139a324990a227b232e857f53f = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b3abef996f5b41949226a3f9a34402cf = L.marker(
                [40.57542419433594,-73.9908676147461],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_6fd12a2ea8134298a02f736e06caa318 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_772d2bddeae94a04ba49352e70bb6b23 = L.marker(
                [40.93547821044922,-73.78217315673827],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1fa568918e854892ba0fc44777921940 = L.marker(
                [40.69514083862305,-74.17729949951173],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ec640a62c7c04c849bb37975424cd464 = L.marker(
                [40.92499542236328,-73.88568115234375],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_394ca4f217454e4daec969d4befcf01a = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_957682636d6b4b73b1f284508472a695 = L.marker(
                [40.69075393676758,-74.17747497558594],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_67928aa7dd7f41f9bc26295de35d0f9c = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_2a99e72d366245b2ae444ab07b6b0dfc = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_6f04f1995d944fe8b998df83d8af0cf9 = L.marker(
                [40.76496887207031,-73.53260040283203],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_10be2c22b5304402a7927a26832058be = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a03343de1ade4511bfd948ca73b0299e = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b5c3eddd55e648f5b2369dc46d2ce1f7 = L.marker(
                [40.690223693847656,-74.17779541015625],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_51da4eb13dc144198c82320f2611e257 = L.marker(
                [40.571876525878906,-73.86095428466798],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8e8885bb8cbe4f2d943098af4d6ec9d3 = L.marker(
                [40.69536590576172,-74.17774963378906],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a46c627b7b884a3e8c1b6c6323d5d25a = L.marker(
                [40.93607711791992,-73.76426696777344],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_5fc8300469484783a5e0bf55d7a0b4ea = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b5f7f0b074874d79ad88511c96af41ec = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_9ef47f7d2b5c4f779a2f84388a1b87b7 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_5ddcae6043774d99a5f22affbea87786 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_548e9911947a45ae81cc6e929d9aa6ee = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_944aea446975406daaaaa77d1fad00f2 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_2a275232202b484491fe094d4a9ce4f0 = L.marker(
                [40.57545852661133,-73.9498062133789],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c21f36fcfb754493b964502016623157 = L.marker(
                [40.69754028320313,-73.69508361816406],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_557cae0fe13b40f18170c07f5f7c68b2 = L.marker(
                [40.67799758911133,-73.60989379882812],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_472814a7658344b7ad7faf053b179917 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_5f9b159d112a43e88b8e70844c6a5005 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_707576361d7b4f8e8559db1d504f06f3 = L.marker(
                [40.690345764160156,-74.17765808105469],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_2518a5b9ea48455abbf0d0a74bdf3739 = L.marker(
                [40.6943473815918,-74.17674255371094],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_653c1faae7a04ae4a4b5f41d88eb8df9 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_38c85ecb54794da98a78f04b0ff3d127 = L.marker(
                [41.06118392944336,-73.83699035644531],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_0166fab210be46fe9fa8f9b043fc5173 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1eec8600f35342d1a35b1e2460cfa301 = L.marker(
                [40.69504928588867,-74.17715454101562],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_3f974a3eada649aca5d21b4bc9e2406f = L.marker(
                [40.6408576965332,-74.2877655029297],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_5df65884694643de96d0196a7e4d4434 = L.marker(
                [40.71596908569336,-73.54425048828125],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_2cfbf21a0cba4f0f8d2651ea998fc9df = L.marker(
                [40.76482772827149,-73.53260803222656],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b9dec3aa79b043d2b226519a907a3557 = L.marker(
                [40.68770599365234,-74.181640625],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_9026b58fb0c94dc28c0892e434d2459b = L.marker(
                [40.76496887207031,-73.53260040283203],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8a4e2bd3d5e7473b863cbbe44e20f38e = L.marker(
                [40.7134895324707,-74.15735626220702],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_68db5c3d238542cd86983c25f2ae8c19 = L.marker(
                [40.7009162902832,-73.66770172119139],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8cbc9fdf3fdc44618bd80b53bf1acc47 = L.marker(
                [40.6880111694336,-74.18316650390625],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ae0ad612d93e4544a812d2ea767322bd = L.marker(
                [40.6943244934082,-74.17671203613281],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_3a6522d6db494aaf82efb8fcf71ca4fc = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_cdf536c9fcc74be7a7b251fed959d999 = L.marker(
                [40.69568252563477,-74.17855072021484],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_27227164459a4ec3bc30459313fbf6ff = L.marker(
                [40.6906623840332,-74.17745208740234],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f86a4295a0a14abc9f906c4293c94215 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_0ecef59065114c8092624cf53995a3a0 = L.marker(
                [40.68932342529297,-74.18426513671875],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b6f9c02dcd104980900bb6b3930d74c2 = L.marker(
                [40.648292541503906,-74.34429168701173],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_5a8f553017f04b04bafe3b876f30c475 = L.marker(
                [40.68769073486328,-74.1815185546875],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ff34ef2ec15c442d90dc893d1a0d9419 = L.marker(
                [40.5760612487793,-73.9687271118164],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_28bae051aee0431cb3efd32f0485ea10 = L.marker(
                [40.69916915893555,-74.1855926513672],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_68f788a4f7854ca6bfd357e459a02488 = L.marker(
                [40.69489669799805,-74.17705535888672],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e3239fbb616b4508b62efe67a6c444c9 = L.marker(
                [40.68770980834961,-74.18142700195312],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_3b16937659434e5b9c2ed30116e69a39 = L.marker(
                [40.87004852294922,-74.45069122314453],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_fe5f08bbd9a1416c92538d9fd2ee7da0 = L.marker(
                [40.68775939941406,-74.18153381347656],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_44c84054d4b94ba79b4a0b5a4ac3c4c7 = L.marker(
                [48.92368698120117,-86.73170471191406],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_3162981467e248eeb498e7277310bb80 = L.marker(
                [40.810859680175774,-73.56905364990234],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_5eefcb570448457e861ec2c2ecd47479 = L.marker(
                [40.91845703125,-74.12361907958984],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_303e4b7d72804650aa069dd1055816a0 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_7e11f8cb3bd94c1fa5b9ccfc07e95839 = L.marker(
                [40.69020462036133,-74.17770385742188],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1b23c41848954e40a11923252db36f33 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_951f4269b9e34878ac93a9b5c3d65f74 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e8448644b32f4e729ed9aaef11805a28 = L.marker(
                [40.68994522094727,-74.1779556274414],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_396431877a5a45d2ad8311440b46b826 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_231a9589191e4322b2660aea8e3e49f0 = L.marker(
                [40.757720947265625,-74.23133087158203],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_3b7b877ea626452ca1fd23d40efdfc8f = L.marker(
                [40.56372833251953,-73.97843933105469],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_bb04736c771c4fe8bb28ec87498b4f97 = L.marker(
                [41.03627014160156,-73.76044464111328],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_74833c9cd14b425589b683eec369f744 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_579fb3b6de8843bc91271c1aa14ebc57 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_3793ae85bb3a4cee9c3adbcfffe12b5b = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f5fe491b3ade43009c0f72e846a5c29c = L.marker(
                [40.69314193725585,-74.1863021850586],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1257bd2ae8ea4f34b905a51555b9acf9 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_7814c9fe0dbc43fba094f1ba007c8e4a = L.marker(
                [40.57603454589844,-73.95172882080078],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_befeb3187ffa4ae58a81c89b7459efde = L.marker(
                [40.68769836425781,-74.1817626953125],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_752029fba6444763924145c4f9e07ebb = L.marker(
                [40.69314193725585,-74.1863021850586],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1f58dc093af049308769942c120b5e37 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_5d58a8722a4f43359fa8236509a03e83 = L.marker(
                [40.6899642944336,-74.17789459228516],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_4af0847855c049e8835e04261b771135 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_12d22343ee8b41898f5bb189af3fa758 = L.marker(
                [40.71163940429688,-74.17940521240234],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c987e293393d4f1ba92c1941f552ea07 = L.marker(
                [40.69491958618164,-74.1771011352539],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c951d890a3304ba3ad347a04d103c2e6 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_4a7258942f364cb5a0c6e97e8a9c6642 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_49c05b0d9eaa43a18230ed4fad6e4904 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_6b83dac616e44eed9718ba57e1a71c9b = L.marker(
                [40.94634628295898,-73.8775634765625],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_3a0c4ef6d3594485976344e1712ae76f = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_688e3be116d04fb0ad246415c7f1692c = L.marker(
                [40.73989868164063,-73.69877624511719],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_7cdf71cbf7d84a71a368c2c37513b509 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_fe6f80c15f1c42d88755cc52447127bb = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_42e2bf5efd334d3b932fdd42d48635f6 = L.marker(
                [40.621971130371094,-74.26590728759766],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_d45c0fc3f2d041eaa12a230679208b27 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a4730861673a4a5bae9005876fac6578 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_25c672aefe99463fb36585dda90c61c2 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_cb01c955be534891b4d0b9401977edac = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_16d7736d4b5c465b8b2302f7befa3ecc = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_df24c1da778b4a4681fd741118a4486a = L.marker(
                [40.690425872802734,-74.17755889892578],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_df084c6f14b648f2bbf8247232647daf = L.marker(
                [40.69050598144531,-74.1775894165039],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f61778950e3b485aa5295bf49fdc0f86 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_97c2bcfa08524cb79c940a0d222cfe6c = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_18af8371dc9f401baf2806833654ef17 = L.marker(
                [40.72400665283203,-73.58769989013672],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_223b5b68341a4d13a560cc98e2d7be9e = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_0ef4ab37ac7b4ad7bea034a0374fd84b = L.marker(
                [40.68778610229492,-74.1827392578125],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8097071b1fb7424e8bf614ca913b7739 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_fbf76e43a91f40af8309847374858948 = L.marker(
                [40.6951904296875,-74.17732238769531],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_3466a1160e20436a8263e8c26d53aa5c = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f1e290d8c6bf4c259d9320d68cf7a53c = L.marker(
                [40.68799209594727,-74.1832275390625],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_7137941b311d4c12abe531c478907e0a = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_47181048964745c385f1f5bb30c54221 = L.marker(
                [40.69541549682617,-74.17788696289062],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_33c1b6056c6946128393f466b4679986 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_89d5500a03f5424693410577f80e5aa6 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_5c00747fdddc4e1791301aeab1e388dc = L.marker(
                [40.57373046875,-73.85310363769531],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_edefcc1dd3bf4b23b565d70c07834541 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_eb3aa9ae7d2b47a7ae09884ca869ba0f = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_bb7bfb95716844639e37dc281e9a2c26 = L.marker(
                [40.69510269165039,-74.17723083496094],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e95011de4ce64a469a5d582cdb6dbfbd = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_54988783f08c44c4ba25b7f87ca98eaf = L.marker(
                [40.69207763671875,-74.17701721191406],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1bc41bd1ffc34892bb6bc292e89aea92 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_2e6a2098e856470282bec44865041841 = L.marker(
                [40.69525146484375,-74.17755889892578],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_30f84b966c8e49239121246a693caae1 = L.marker(
                [40.69058609008789,-74.17752075195312],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_72443d623ee249e59bf312012f74bc89 = L.marker(
                [40.930084228515625,-73.78417205810547],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_d0a6aa6d8fe64a9c81c510c996284726 = L.marker(
                [40.68769073486328,-74.18166351318358],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a2874e7ffa8b4c2898b9f97dfca086d6 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_6a520277124341aa90bd1989fe4e1d57 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b947a3b1a0ce41c9961ffc3b8b919a83 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c3945e2cc853476c9df2e9bb9efb1164 = L.marker(
                [40.69522476196289,-74.17742919921875],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_abc8b90201f14d4e94c7f0181e5c3d55 = L.marker(
                [40.65894317626953,-73.69688415527344],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_2a4a0f8ac1544190916c8c23d0bfb2d1 = L.marker(
                [40.69512176513672,-74.17726135253906],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_8c4c3426e3f24d80bf2696abf38f729a = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_fc61dc6ebdde45ea90fbec299819405c = L.marker(
                [40.692142486572266,-74.1812515258789],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_136188a663ca408da8aff46388518fd5 = L.marker(
                [40.69012069702149,-74.17781066894531],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_9e1d44e970714d23abcd822edc571ab8 = L.marker(
                [40.695125579833984,-74.17727661132812],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_485a1b14ea684ab8922838b18a4b7a66 = L.marker(
                [40.76482772827149,-73.53260803222656],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_7f4ce20f486e4aefbf7cce2a6d9616f6 = L.marker(
                [40.784175872802734,-73.52144622802734],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_e9da199d444a4bed8144ecb4e3c540d8 = L.marker(
                [40.6903190612793,-74.17767333984375],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_98c63d34e64448bfb50ac2e27d1122a5 = L.marker(
                [40.68982315063477,-74.17810821533203],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a0a5648187be4d9eb6628b4a95e49d07 = L.marker(
                [40.6905403137207,-74.1775665283203],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_3fb31ff9bc1f4977a9a44677e4009942 = L.marker(
                [40.69072341918945,-74.17745208740234],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1c8cd6ba61c140ac8fc56d077dbdf857 = L.marker(
                [40.695178985595696,-74.17767333984375],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_d791662fc6f84185a5dc81f24d285b5c = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_6c3abecbf7514b92be789750f8ea7578 = L.marker(
                [40.69011688232422,-74.17814636230467],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_1634c3aa24ac437c9f5eac8ed2f58938 = L.marker(
                [40.6537971496582,-74.35639953613281],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_9f4789dd27ca4652bdee7282d05426c7 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a1b1f2a3879d463da58813a388f6d664 = L.marker(
                [40.74110794067383,-73.58686828613281],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ce1cbfae17ae44218349f6ca654e326e = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a19b939189504a4fa2859f9f3919550d = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_37dae46788994a55b4d499ffacd19b3b = L.marker(
                [40.68185806274415,-73.57048797607422],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_640ad33cca2243698255e4dd348e77c9 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ece16c611d114c5c97961a9c8709fb9b = L.marker(
                [40.78361511230469,-73.42215728759766],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_17aa32ff1c5f471fa6584b5a35f4396c = L.marker(
                [40.69477844238281,-74.17703247070312],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_cebe65a1e975459d9ff3bf534d9b35b6 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_b7c15a1bc8ff4b74a36b03d0ff68f71d = L.marker(
                [40.67937088012695,-73.70021057128906],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_30c212cfbac347e5a986c30391244e24 = L.marker(
                [40.69038772583008,-74.17830657958984],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_86cd6f417a0f49248595543c058ca9f0 = L.marker(
                [40.68755722045898,-74.18163299560547],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_2902a99fdecc41d28f045a405034bdc8 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ca01e70744be4c8f9c93f4811da7544b = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_5e8566f478904e298474641b72106f54 = L.marker(
                [40.687896728515625,-74.18158721923827],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_79ef71cbeec3479b84863213c00e6492 = L.marker(
                [40.69513320922852,-74.1773681640625],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_44ba27b87a6d4333b7088247fa96fbbd = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_9b245ee521f44f09a0b17ed1a63558ff = L.marker(
                [40.65277099609375,-74.17278289794923],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_158bb57794f94bccaf39869ca35b5f88 = L.marker(
                [40.69050216674805,-74.1776123046875],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_c4ff46c42eac4c73b9f8abe6fde418b4 = L.marker(
                [40.69457626342773,-74.17684173583984],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_dba1845ed6cb4fbeaa493ac256e79dd5 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_29b0cc6642b043c3b8d1c9bf4e7648d5 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_369fd9e294c74ddfa209f179bc56b250 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_5c242987743d453e9e67aafdda5e0511 = L.marker(
                [40.71336364746094,-73.59330749511719],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_74417f0f9ae5454da55ec5528a299be7 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_d3a97e0d2f8642babb75004c82bc20f4 = L.marker(
                [40.69499588012695,-74.1770248413086],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_09a5161270584a78943288255fef108d = L.marker(
                [40.68773651123047,-74.18228912353516],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_591f654aa6494648ae7803d31c6eb983 = L.marker(
                [40.69014358520508,-74.17784118652342],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_ff5cbe5b69d84bc39bba06ad3e984538 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_f3e071b688604240946af0fe0579324a = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_a0ac80b422394b66a266a330afbe3be3 = L.marker(
                [40.800559997558594,-74.1985626220703],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
    

            var marker_7d690666e7e94c86ae30466449a8a448 = L.marker(
                [0.0,0.0],
                {
                    icon: new L.Icon.Default()
                    }
                )
                .addTo(map_5d87149c73ac45bb9f7cba04c6707063);
            
</script>
