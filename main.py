import networkx as nx
import requests
from datetime import datetime
from math import sqrt
from geopy.distance import great_circle
import pprint
import pickle
import random
import operator

REAL = False
pp = pprint.PrettyPrinter(indent=4)
CHECKIN_PICKLE = 'checkins_new.pickle'
CHECKIN_PICKLE_2 = 'checkins_new_part_2.pickle'
#CHECKIN_PICKLE = 'checkins_nonAPI_able.pickle'
EDGE_PICKLE = 'edge.pickle'
#CHECKIN_TRAIN = 'checkin_usersGPS_largcomp_index.csv'
CHECKIN_TRAIN = 'checkin_usersGPS_largcomp_training.csv'
CHECKIN_REAL = 'checkin_usersGPS_largcomp_index.csv'
EDGE_TRAIN = 'usersGPS_larg_comp_training.txt'

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
        CHECKIN_ORIGINAL_FILE = CHECKIN_REAL if REAL else CHECKIN_TRAIN
        with open(CHECKIN_ORIGINAL_FILE, 'r') as f:
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

# most recent only
def getCheckins2(saved=False):
    if saved:
        with open(CHECKIN_PICKLE_2, 'rb') as p:
            checkins = pickle.load(p)
            return checkins
    else:
        checkins = {}
        CHECKIN_ORIGINAL_FILE = CHECKIN_REAL if REAL else CHECKIN_TRAIN
        with open(CHECKIN_ORIGINAL_FILE, 'r') as f:
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
    miles_threshold = 29
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

def sortNodesSaved(IDs, checkins, G, endLocs, degree_dict_param):
    closest_dist_dict = {}
    degree_dict = {}
    allInfinity = True
    for ID in IDs:
        closest_dist = getClosestDist(ID, checkins, G, endLocs)
        if closest_dist == float('inf'):
            degree_dict[ID] = G.degree(ID) if not degree_dict_param else degree_dict_param[ID]
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
    start_vals = [2, 3]
    # number of iterations to shorten
    T = 30
    for t in range(T):
        start_val_idx = t % 2
        start_idx = start_vals[start_val_idx]
        for i in range(start_idx, len(path)-1, 2):
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
                mutual_loc = getLoc(mutual, checkins)
                dist = getGeoDist(loc1, mutual_loc) + getGeoDist(mutual_loc, loc2)
                if dist < minMutualDist:
                    minMutualDist = dist
                    minMutual = mutual
            path[i-1] = minMutual
        print("currPath: ", path)
        print("currLen: ", findPathLength(path, checkins))

    return path

def getNeighbors(currNodeID, testcaseID):
    URL = 'https://6io70nu9pi.execute-api.us-east-1.amazonaws.com/smallworld/neighbors?node=%d&uni=bll2121&testcaseID=%d' % (currNodeID, testcaseID)
    r = requests.get(URL)
    response = r.json()
    neighbors_response = response['neighbors']
    neighbors = {int(x): neighbors_response[x] for x in neighbors_response.keys()}
    return neighbors

def findPath(checkins, G, startId=None, endId=None, testcaseID):
    if startId == None or endId == None: return None
    actual_run = True if G == None else False
    print(startId, endId)

    degree_dict = {}
    queries = 0
    endLocs = checkins[endId]
    if actual_run:
        endNeighborsDict = getNeighbors(endId, testcaseID)
        endNeighborList = [ID for ID in endNeighborsDict]
    else:
        endNeighborList = G.neighbors(endId)
    queries += 1
    endNeighbors = set(endNeighborList)
    if actual_run:
        endNeighborDegrees = {
                ID: endNeighborsDict[ID][0] for ID in endNeighborsDict
        }
        degree_dict.update(endNeighborDegrees)
    else:
        endNeighborDegrees = {ID: G.degree(ID) for ID in endNeighbors}
    sortedNeighborsDeg = sorted(endNeighborDegrees.items(),
            key=operator.itemgetter(1))
    # find the best of the end neighbors to continue adding more nodes for
    # searching
    maxEndNeighbor = sortedNeighborsDeg[-1]

    # add some more neighbors
    maxEndNeighborId = maxEndNeighbor[0]
    if actual_run:
        endNeighborNeighborsDict = getNeighbors(maxEndNeighborId, testcaseID)
        endNeighborNeighborsList = [ID for ID in endNeighborNeighborsDict]
        endNeighborNeighborDegrees = {
                ID: endNeighborNeighborsDict[ID][0] for ID in endNeighborNeighborsDict
        }
        degree_dict.update(endNeighborNeighborDegrees)

    else:
        endNeighborNeighborsList = G.neighbors(maxEndNeighborId)

    endNeighborNeighbors = set(endNeighborNeighborsList)
    queries += 1

    # keeping track of all the nodes we've seen so far
    visited = set()
    stack = [startId]
    back = {startId: None}
    # node1, node2 = [int(x) for x in line[:-1].split()]
    # G.add_edge(node1, node2)

    while stack:
        curr = stack.pop()
        if curr == endId:
            break

        if curr not in visited:
            visited.add(curr)
            # if we are doing the API
#            r = requests.get('https://6io70nu9pi.execute-api.us-east-1.amazonaws.com/smallworld/neighbors?node=%s&uni=bll2121&testcaseID=1' % (curr))
#            response = r.json()
#            neighbors = response['neighbors']
#            unvisitedNeighbors = set()

            if actual_run:
                neighDict = getNeighbors(curr, testcaseID)
                neighDegreeDict = {
                    ID: neighDict[ID][0] for ID in neighDict
                }
                neighList = [ID for ID in neighDict]
                degree_dict.update(neighDegreeDict)
            else:
                neighList = G.neighbors(curr)
            queries += 1
            neighs = set(neighList)

            unvisitedNeighbors = neighs - visited

            # we found the neighbor of the destination node, so we are
            # basically done
            possibleOverlap = unvisitedNeighbors.intersection(endNeighbors)
            if len(possibleOverlap) > 0:
                # pick any of the points in the intersection and use that to
                # get to the target node
                shortcut = next(iter(possibleOverlap))
                back[shortcut] = curr
                back[endId] = shortcut
                break

            furtherOverlap = unvisitedNeighbors.intersection(endNeighborNeighbors)
            if len(furtherOverlap) > 0:
                # same thing as before but now it's one removed
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

            if len(unvisitedNeighbors) > 0:
                truncateAmount = 3
                sortedNeighbors = sortNodesSaved(unvisitedNeighbors, checkins,
                        G, endLocs, degree_dict)
                stack.extend(sortedNeighbors[-truncateAmount:])

    print(back)
    path = []
    currId = endId
    while currId != None:
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
    part = 1
    if part == 1:
        saved = True
        checkins = getCheckins(saved)
        G = getGraph(saved)
        #G = None
#        if REAL:
#        # for the API
#            r = requests.get('https://6io70nu9pi.execute-api.us-east-1.amazonaws.com/smallworld/start/1')
#            response = r.json()
#            startId = response['source node']
#            endId = response['target node']

        random.seed()
        startId = random.randint(0, 47708)
        endId = random.randint(0, 47708)
        #startId = None
        #endId = None

        path = findPath(checkins, G, startId, endId)
        #T = 1
        #for t in range(T):
        #    random.seed()
        #    startId = 0
        #    endId = 2
        ##    startId = random.randint(0, 47708)
        ##    endId = random.randint(0, 47708)
        #    path = findPath(checkins, G, startId, endId)

    elif part == 2:
        saved = True
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
            path = findPath2(checkins2, G, originalPath)
            print(path)
            print("new geolength is: ", findPathLength(path, checkins2))
