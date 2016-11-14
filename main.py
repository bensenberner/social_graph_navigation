import networkx as nx
import requests
from datetime import datetime
from math import sqrt
from geopy.distance import great_circle
import pprint
import pickle
import random
import operator

pp = pprint.PrettyPrinter(indent=4)
CHECKIN_PICKLE = 'checkins_new.pickle'
CHECKIN_PICKLE_2 = 'checkins_new_part_2.pickle'
#CHECKIN_PICKLE = 'checkins_nonAPI_able.pickle'
EDGE_PICKLE = 'edge.pickle'
#CHECKIN_TRAIN = 'checkin_usersGPS_largcomp_index.csv'
CHECKIN_TRAIN = 'checkin_usersGPS_largcomp_training.csv'
EDGE_TRAIN = 'usersGPS_larg_comp_training.txt'

def addToAverage(avg, size, val):
    return (size * avg + val) / (size + 1)

# build edges
def getGraph(saved=False):
    if not saved:
        G = nx.Graph()
        with open(EDGE_TRAIN, 'r') as f:
            for line in f:
                node1, node2 = [int(x) for x in line[:-1].split()]
                G.add_edge(node1, node2)
        with open(EDGE_PICKLE, 'wb') as p:
            pickle.dump(G, p, pickle.HIGHEST_PROTOCOL)
    else:
        with open(EDGE_PICKLE, 'rb') as p:
            G = pickle.load(p)
    return G

# build checkins
def getCheckins(saved=False):
    if saved:
        with open(CHECKIN_PICKLE, 'rb') as p:
            checkins = pickle.load(p)
            return checkins
    else:
        checkins = {}
        with open(CHECKIN_TRAIN, 'r') as f:
            for line in f:
                curr_data = line[:-1].split(',')
                ID, lat_val, long_val, timestamp = curr_data
                ID = int(float(ID))
                ts = float(timestamp)
                lat_val = float(lat_val)
                long_val = float(long_val)

                if ID not in checkins:
                    checkins[ID] = [
                            [lat_val, long_val, ts]
                    ]
                else:
                    checkins[ID].append(
                            [lat_val, long_val, ts]
                    )

        # save the object for the future
        with open(CHECKIN_PICKLE, 'wb') as p:
            pickle.dump(checkins, p, pickle.HIGHEST_PROTOCOL)

        return checkins

def getCheckins2(saved=False):
    if saved:
        with open(CHECKIN_PICKLE_2, 'rb') as p:
            checkins = pickle.load(p)
            return checkins
    else:
        checkins = {}
        with open(CHECKIN_TRAIN, 'r') as f:
            for line in f:
                curr_data = line[:-1].split(',')
                ID, lat_val, long_val, timestamp = curr_data
                ID = int(float(ID))
                ts = float(timestamp)
                lat_val = float(lat_val)
                long_val = float(long_val)

                if ID not in checkins:
                    checkins[ID] = [lat_val, long_val, ts]
                elif ts > checkins[ID][2]:
                    checkins[ID] = [lat_val, long_val, ts]

        # save the object for the future
        with open(CHECKIN_PICKLE_2, 'wb') as p:
            pickle.dump(checkins, p, pickle.HIGHEST_PROTOCOL)

        return checkins

def getClosestDist(ID, checkins, G, endLocs):
    # TODO: fiddle around
    #days = 1
    #hours = 24 * days
    #hours = 0.2
    #minutes = 60 * hours
    minutes = 200
    # it all boils down to using seconds
    seconds_threshold = 60 * minutes
    miles_threshold = 25
    minDist = float('inf')
    for currCheckin in checkins[ID]:
        for endCheckin in endLocs:
            currCheckinTime = currCheckin[2]
            endCheckinTime = endCheckin[2]
            # see if the current pair of checkins happened within the seconds
            # threshold
            if abs(currCheckinTime - endCheckinTime) < seconds_threshold:
                currLoc = (currCheckin[0], currCheckin[1])
                endLoc = (endCheckin[0], endCheckin[1])
                dist = great_circle(currLoc, endLoc).miles
                # see if the curr pair of checkins occurred with the spatial
                # threshold
                if dist < miles_threshold:
                    minDist = min(minDist, dist)
    return minDist

def sortNodesSaved(IDs, checkins, G, endLocs):
    closest_dist_dict = {}
    degree_dict = {}
    allInfinity = True
    for ID in IDs:
        closest_dist = getClosestDist(ID, checkins, G, endLocs)
        if closest_dist == float('inf'):
            #degree_dict[ID] = G.degree(ID) / G.
            degree_dict[ID] = G.degree(ID)
        else:
            allInfinity = False
            closest_dist_dict[ID] = closest_dist

    actualDict = degree_dict if allInfinity else closest_dist_dict
    shouldReverse = not allInfinity
    sortedDict = sorted(actualDict.items(), key=operator.itemgetter(1),
            reverse=shouldReverse)
    sortedNodesBySimilarity = [x[0] for x in sortedDict]
    return sortedNodesBySimilarity

# TODO: this is the old way
#def sortNodesSaved(IDs, checkins, G, endLoc)
#    weightDict = {}
#    distDict = {}
#    degDict = {}
#    #clustDict = {}
#    sumDists = 0
#    sumDegrees = 0
#    #sumCluster = 0
#    for ID in IDs:
#        dist = getDistanceID(ID, checkins, endLoc)
#        distDict[ID] = dist
#        sumDists += dist
#
#        degree = G.degree(ID)
#        degDict[ID] = degree
#        sumDegrees += degree
#
#        # cluster = nx.clustering(G, ID)
#        # clustDict[ID] = cluster
#        # sumCluster += cluster
#
#        #weightDict[ID] = dist + 0.8 * degree
#        #sumDists += dist
#    for ID in IDs:
#        normDist = distDict[ID] / sumDists
#        # TODO: update with deg
#        normDeg = degDict[ID] / sumDegrees
#        #normDeg = 0
#        #normClust = clustDict[ID] / sumCluster
#        #weightDict[ID] = normDist + 1.15 * normDeg
#        #weightDict[ID] = normDist + 0.75 * normDeg
#        weightDict[ID] = normDist + normDeg
#
#    #pp.pprint(weightDict)
#    return [x[0] for x in sorted(weightDict.items(),
#        key=operator.itemgetter(1))]
def getLoc(ID, checkins):
    checkin = checkins[ID]
    return (checkin[0], checkin[1])

def getGeoDist(loc1, loc2):
    return great_circle(loc1, loc2).miles

def findPathLength(path, checkins):
    length = 0
    for i in range(1, len(path)):
        loc1 = getLoc(path[i-1], checkins)
        loc2 = getLoc(path[i], checkins)
        dist = getGeoDist(loc1, loc2)
        length += dist
    return length

def findPath2(checkins, G, path):
    for i in range(2, len(path)-1, 2):
        ID1 = path[i-2]
        loc1 = getLoc(ID1, checkins)
        ID2 = path[i]
        loc2 = getLoc(ID2, checkins)

        # find the intersection of the neighbors
        neighbors1 = set(G.neighbors(ID1))
        neighbors2 = set(G.neighbors(ID1))
        neighbor_intersection = neighbors1.intersection(neighbors2)
        minMutualDist = float('inf')
        minMutual = None
        # pick the best mutual in place of what is already there
        for mutual in neighbor_intersection:
            # compute the distance from node1 --> mutual --> node2, minimize
            mutual_loc = getLoc(mutual)
            dist = getGeoDist(loc1, mutual_loc) + getGeoDist(mutual_loc, loc2)
            if dist < minMutualDist:
                minMutualDist = dist
                minMutual = mutual
        path[i-1] = minMutual
    return path

def findPath(checkins, G, startId=None, endId=None):
    print(startId, endId)
    queries = 0
    endLocs = checkins[endId]
    endNeighbors = set(G.neighbors(endId))
    queries += 1
    endNeighborDegrees = {ID: G.degree(ID) for ID in endNeighbors}
    sortedNeighborsDeg = sorted(endNeighborDegrees.items(),
            key=operator.itemgetter(1))
    maxEndNeighbor = sortedNeighborsDeg[-1]

    # add some more neighbors
    maxEndNeighborId = maxEndNeighbor[0]
    queries += 1
    endNeighborNeighbors = set(G.neighbors(maxEndNeighborId))

    visited = set()
    stack = [startId]
    back = {startId: None}

    while stack:
        curr = stack.pop()
        if curr == endId:
            break

        if curr not in visited:
            visited.add(curr)
            queries += 1
            # if we are doing the API
#            r = requests.get('https://6io70nu9pi.execute-api.us-east-1.amazonaws.com/smallworld/neighbors?node=%s&uni=bll2121&testcaseID=1' % (curr))
#            response = r.json()
#            neighbors = response['neighbors']
#            unvisitedNeighbors = set()

            neighs = set(G.neighbors(curr))
            unvisitedNeighbors = neighs - visited

            # we found the neighbor of the destination node, so we are
            # basically done
            possibleOverlap = unvisitedNeighbors.intersection(endNeighbors)
            if len(possibleOverlap) > 0:
                shortcut = next(iter(possibleOverlap))
                back[shortcut] = curr
                back[endId] = shortcut
                break

            furtherOverlap = unvisitedNeighbors.intersection(endNeighborNeighbors)
            if len(furtherOverlap) > 0:
                shortcut = next(iter(furtherOverlap))
                back[shortcut] = curr
                back[maxEndNeighborId] = shortcut
                back[endId] = maxEndNeighborId
                break

            # set the backpointer
            for neighbor in unvisitedNeighbors:
                back[neighbor] = curr

            # we have found the end node so here we exit the search
            if endId in unvisitedNeighbors:
                break

            # for part 1
            # TODO: old way
            if len(unvisitedNeighbors) > 0:
                truncateAmount = 3
                sortedNeighbors = sortNodesSaved(unvisitedNeighbors, checkins,
                        G, endLocs)
                stack.extend(sortedNeighbors[-truncateAmount:])

    path = []
    currId = endId
    while currId:
        path.append(currId)
        if not currId in back:
            print("Could not find ", endId, currId)
            break
        currId = back[currId]
    path.reverse()

    print("queries", queries)
    shortest_path = nx.shortest_path(G, startId, endId)
    print("shortest path", shortest_path)
    print("path: ", path)

    return path

if __name__ == "__main__":
    part = 2
    if part == 1:
        saved = True
        checkins = getCheckins(saved)
        G = getGraph(saved)

        # for the API
        # r = requests.get('https://6io70nu9pi.execute-api.us-east-1.amazonaws.com/smallworld/start/1')
        # response = r.json()
        # startId = response['source node']
        # endId = response['target node']

        for t in range(100):
            random.seed()
            startId = random.randint(0, 47708)
            endId = random.randint(0, 47708)
            #minTimeA = min(x[2] for x in checkins[15299])
            #minTimeB = min(x[2] for x in checkins[44742])
            #a = [[x[0], x[1], x[2] - minTimeA] for x in checkins[15299]]
            #b = [[x[0], x[1], x[2] - minTimeB] for x in checkins[44742]]
            # for node in nx.shortest_path(G, startId, endId):
                # pp.pprint(G.neighbors(node))
            #print('all the end neighbors')
            #pp.pprint(G.neighbors(endId))
            #print('#####################')
            #pp.pprint(a)
            #pp.pprint(b)
            #pp.pprint(nx.shortest_path(G, startId, endId))
            path = findPath(checkins, G, startId, endId)

    elif part == 2:
        saved = False
        checkins = getCheckins(saved)
        checkins2 = getCheckins2(saved)
        G = getGraph(saved)
        for t in range(1):
            random.seed()
            startId = random.randint(0, 47708)
            endId = random.randint(0, 47708)
            originalPath = findPath(checkins, G, startId, endId)
            print(originalPath)
            print("geoLengthIs: ", findPathLength(originalPath, checkins2))
            path = findPath2(checkins2, G, path)
            print(path)
            print("new geolength is: ", findPathLength(path, checkins2))
