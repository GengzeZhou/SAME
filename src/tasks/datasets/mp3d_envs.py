import numpy as np
import math
import networkx as nx
import json
import os
import math
from tqdm import tqdm
import msgpack
# import msgpack_numpy
# msgpack_numpy.patch()


def angle_feature(heading, elevation, angle_feat_size):
    return np.array(
        [math.sin(heading), math.cos(heading),
         math.sin(elevation), math.cos(elevation)] * (angle_feat_size // 4),
        dtype=np.float32)


def get_point_angle_feature(angle_feat_size, baseViewId=0):
    feature = np.empty((36, angle_feat_size), np.float32)
    base_heading = (baseViewId % 12) * math.radians(30)
    base_elevation = (baseViewId // 12 - 1) * math.radians(30)

    headings = [0.0, 0.5235987755982988, 1.0471975511965976, 1.5707963267948966, 2.0943951023931953, 2.617993877991494, 3.141592653589793, 3.665191429188092, 4.1887902047863905, 4.71238898038469, 5.235987755982988, 5.759586531581287, 0.0, 0.5235987755982988, 1.0471975511965976, 1.5707963267948966, 2.0943951023931953, 2.617993877991494, 3.141592653589793, 3.665191429188092, 4.1887902047863905, 4.71238898038469, 5.235987755982988, 5.759586531581287, 0.0, 0.5235987755982988, 1.0471975511965976, 1.5707963267948966, 2.0943951023931953, 2.617993877991494, 3.141592653589793, 3.665191429188092, 4.1887902047863905, 4.71238898038469, 5.235987755982988, 5.759586531581287]
    elevations = [-0.5235987755982988, -0.5235987755982988, -0.5235987755982988, -0.5235987755982988, -0.5235987755982988, -0.5235987755982988, -0.5235987755982988, -0.5235987755982988, -0.5235987755982988, -0.5235987755982988, -0.5235987755982988, -0.5235987755982988, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5235987755982988, 0.5235987755982988, 0.5235987755982988, 0.5235987755982988, 0.5235987755982988, 0.5235987755982988, 0.5235987755982988, 0.5235987755982988, 0.5235987755982988, 0.5235987755982988, 0.5235987755982988, 0.5235987755982988]

    for ix, (heading, elevation) in enumerate(zip(headings, elevations)):
        heading = heading - base_heading
        elevation = elevation - base_elevation
        feature[ix, :] = angle_feature(heading, elevation, angle_feat_size)
    return feature


def get_all_point_angle_feature(angle_feat_size):
    return [get_point_angle_feature(angle_feat_size, baseViewId) for baseViewId in range(36)]


def load_graphs(connectivity_dir, scans):
    """
    load graph from scan,
    Store the graph {scan_id: graph} in graphs
    Store the shortest path {scan_id: {view_id_x: {view_id_y: [path]} } } in paths
    Store the distances in distances. (Structure see above)
    Load connectivity graph for each scan, useful for reasoning about shortest paths
    :return: None
    """
    graphs = load_nav_graphs(connectivity_dir, scans)
    shortest_paths = {}
    graphs_bar = tqdm(
        graphs.items(),
        desc='Computing shortest paths',
    )
    for scan, G in graphs_bar:  # compute all shortest paths
        shortest_paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
    shortest_distances = {}
    graphs_bar = tqdm(
        graphs.items(),
        desc='Computing shortest distances',
    )
    for scan, G in graphs_bar:  # compute all shortest paths
        shortest_distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))
    
    return graphs, shortest_paths, shortest_distances


def load_nav_graphs(connectivity_dir, scans):
    ''' Load connectivity graph for each scan '''

    def distance(pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pose'][3] - pose2['pose'][3]) ** 2 \
                + (pose1['pose'][7] - pose2['pose'][7]) ** 2 \
                + (pose1['pose'][11] - pose2['pose'][11]) ** 2) ** 0.5

    graphs = {}
    scans = tqdm(
        scans,
        desc='Loading navigation graphs',
    )
    for scan in scans:
        with open(os.path.join(connectivity_dir, '%s_connectivity.json' % scan)) as f:
            G = nx.Graph()
            positions = {}
            data = json.load(f)
            for i, item in enumerate(data):
                if item['included']:
                    for j, conn in enumerate(item['unobstructed']):
                        if conn and data[j]['included']:
                            positions[item['image_id']] = np.array([item['pose'][3],
                                                                    item['pose'][7], item['pose'][11]]);
                            assert data[j]['unobstructed'][i], 'Graph should be undirected'
                            G.add_edge(item['image_id'], data[j]['image_id'], weight=distance(item, data[j]))
            nx.set_node_attributes(G, values=positions, name='position')
            graphs[scan] = G
    return graphs


def normalize_angle(x):
    '''convert radians into (-pi, pi]'''
    pi2 = 2 * math.pi
    x = x % pi2 # [0, 2pi]
    if x > math.pi:
        x = x - pi2
    return x


def convert_heading(x):
    return x % (2 * math.pi) / (2 * math.pi)   # [0, 2pi] -> [0, 1)


def convert_elevation(x):
    return (normalize_angle(x) + math.pi) / (2 * math.pi)   # [0, 2pi] -> [0, 1)


class Simulator(object):
    ''' A simple simulator in Matterport3D environment '''

    def __init__(
            self,
            node_location_dir: str,
            simulation_env: str = None,
        ):
        self.heading = 0
        self.elevation = 0
        self.scan_ID = ''
        self.viewpoint_ID = ''
        self.simulation_env = simulation_env
        with open(node_location_dir, 'r') as f:
            self.node_location = json.load(f)
    
    def _make_id(self, scan_ID, viewpoint_ID):
        return scan_ID + '_' + viewpoint_ID

    def newEpisode(
            self, 
            scan_ID: str, 
            viewpoint_ID: str,
            heading: int, 
            elevation: int = 0,
        ):
        self.heading = heading
        self.elevation = elevation
        self.scan_ID = scan_ID
        self.viewpoint_ID = viewpoint_ID
        # Load navigable dict
        self.location = self.node_location[scan_ID][viewpoint_ID]
        # Calculate viewIndex
        # 0-11: looking down, 12-23: looking at horizon, 24-35: looking up
        self.viewIndex = int((math.degrees(self.heading) + 15) // 30 + 12 * ((math.degrees(self.elevation) + 15) // 30 + 1))

    def getState(self) -> dict:
        self.state = {
            'simulation_env': self.simulation_env,
            'scanId': self.scan_ID,
            'viewpointId': self.viewpoint_ID,
            'viewIndex': self.viewIndex,
            'heading': self.heading,
            'elevation': self.elevation,
            'x': self.location[0],
            'y': self.location[1],
            'z': self.location[2],
        }
        return self.state


class EnvBatch(object):
    ''' A simple wrapper for a batch of MatterSim environments,
        using discretized viewpoints and pretrained features '''

    def __init__(self, node_location_dir, simulation_env, batch_size=1):
        """
        1. Load pretrained image feature
        2. Init the Simulator.
        :param feat_db: The name of file stored the feature.
        :param batch_size:  Used to create the simulator list.
        """
        self.sims = []
        self.simulation_env = simulation_env
        for i in range(batch_size):
            sim = Simulator(node_location_dir=node_location_dir)
            self.sims.append(sim)

    def _make_id(self, scanId, viewpointId):
        return scanId + '_' + viewpointId

    def newEpisodes(self, scanIds, viewpointIds, headings):
        for i, (scanId, viewpointId, heading) in enumerate(zip(scanIds, viewpointIds, headings)):
            self.sims[i].newEpisode(scanId, viewpointId, heading, 0)

    def getStates(self):
        """
        Get list of simulator states.
        """
        agent_states = []
        for i, sim in enumerate(self.sims):
            state = sim.getState()
            state.update(
                {"simulation_env": self.simulation_env,}
            )
            agent_states.append(state)
        return agent_states

    def makeActions(self, actions):
        ''' Take an action using the full state dependent action interface (with batched input).
            Every action element should be an (scanId, viewpointId, heading) tuple. '''
        for i, (scanId, viewpointId, heading) in enumerate(actions):
            self.sims[i].newEpisode(scanId, viewpointId, heading, 0)

