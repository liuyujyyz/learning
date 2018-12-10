import numpy as np
from decorators import timer
from tqdm import tqdm
from geometry import Point, Line, Draw

G = 6.754*10**-11
K = 9*10**9
class Particle:
    def __init__(self, x, y, z, m, e=0, v=(0,0,0)):
        self.position = Point(x, y, z)
        self.weight = m
        self.elec = e
        self.v = Point(*v)

    def force(self, p):
        l = Line(self.position, p.position)
        dist = l.length * 1000
        direct = self.position - p.position
        F = G*self.weight * p.weight / dist**2 - K*self.elec * p.elec / dist**2
        out = (F / dist) * direct 
        return out

    def update(self, f):
        self.position = self.position + self.v
        self.v = self.v + (1/self.weight/1000) * f

class Physics:
    def __init__(self, plist):
        self.plist = plist
        self.time = 0

    def iterate(self):
        q = Draw()
        while True:
            forces = []
            for itemA in self.plist:
                force = Point(0, 0, 0)
                for itemB in self.plist:
                    if itemB == itemA:
                        continue
                    delta = itemB.force(itemA)
                    force = force + delta
                forces.append(force)
            for i in range(len(forces)):
                self.plist[i].update(forces[i])
                q.put_circle(self.plist[i].position.x, self.plist[i].position.y, self.plist[i].position.z)
            h = q.render()
            if h:
                return
            q.clear()

class UnionFind:
    def __init__(self, n):
        self.id = [i for i in range(n)]
        self.size = [1 for i in range(n)]

    def find(self, i):
        p = i
        while p != self.id[p]:
            self.id[p] = self.id[self.id[p]]
            p = self.id[p]
        return p

    def union(self, i, j):
        p = self.find(i)
        q = self.find(j)
        if (p == q):
            return False 
        if self.size[p] < self.size[q]:
            self.id[p] = q
            self.size[p] += self.size[q]
        else:
            self.id[q] = p
            self.size[q] += self.size[p]
        return True


class Graph:
    def __init__(self, num_node, is_direct=True):
        self.edges = np.zeros((num_node, num_node), dtype='float32') - 1
        self.is_direct = is_direct
        self.num_node = num_node
        self.edge_list = []

    def insert_edge(self, i, j, w):
        self.edges[i,j] = w
        self.edge_list.append((i,j,w))
        if not self.is_direct:
            self.edges[j,i] = w
            self.edge_list.append((j,i,w))

    def distance(self, s, t):
        que = [s]
        dist = np.ones((self.num_node,)) * 1000000
        visit = [s]
        dist[s] = 0
        while len(que) > 0:
            now = que.pop(0)
            for i in range(self.num_node):
                if self.edges[now,i] > 0:
                    dist[i] = min(dist[i], dist[now] + self.edges[now, i])
                    if not(i in visit):
                        que.append(i)
                        visit.append(i)
        if t in visit:
            return dist[t]
        else:
            return -1

    @timer
    def is_connected(self):
        que = [0]
        visit = [0]
        while len(que) > 0:
            now = que.pop(0)
            for i in range(self.num_node):
                if self.edges[now, i] > 0 and not(i in visit):
                    que.append(i)
                    visit.append(i)
        if len(visit) != self.num_node:
            return False
        if not self.is_direct:
            return True
        que = [0]
        visit = [0]
        while len(que) > 0:
            now = que.pop(0)
            for i in range(self.num_node):
                if self.edges[i, now] > 0 and not(i in visit):
                    que.append(i)
                    visit.append(i)
        return (len(visit) == self.num_node)

    @timer
    def min_spanning_tree(self):
        if self.is_direct:
            return None
        ufset = UnionFind(self.num_node)
        tree_edge = []
        self.edge_list.sort(key=lambda x:x[2])
        for edge in self.edge_list:
            i, j, w = edge
            union = ufset.union(i, j)
            if union:
                tree_edge.append(edge)
        return tree_edge


class BinaryTreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        self.parent = None 
        self.count = 1

    def show(self):
        for k in self.__dict__:
            print(k, self.__dict__[k])

    @property
    def height(self):
        if self.left is None and self.right is None:
            return 1
        if self.left is None:
            return self.right.height + 1
        if self.right is None:
            return self.left.height + 1
        return max(self.left.height, self.right.height) + 1


class BinaryTree:
    def __init__(self, root_value):
        self.root = BinaryTreeNode(root_value)

    @property
    def size(self):
        count = 0
        l = [self.root]
        while len(l) > 0:
            cur = l.pop(0)
            count += 1
            if cur.left is not None:
                l.append(cur.left)
            if cur.right is not None:
                l.append(cur.right)
        return count 

    def show(self):
        l = [self.root]
        while len(l) > 0:
            cur = l.pop(0)
            tup = [cur.value]
            if cur.left is not None:
                l.append(cur.left)
                tup.append(cur.left.value)
            else:
                tup.append(None)
            if cur.right is not None:
                l.append(cur.right)
                tup.append(cur.right.value)
            else:
                tup.append(None)
            if cur.parent is not None:
                tup.append(cur.parent.value)
            print(tup)
    
    def qianxu(self):
        def qianxu_node(node):
            if node is None:
                return []
            return qianxu_node(node.left) + [node.value] + qianxu_node(node.right)
        return qianxu_node(self.root)

    def insert(self, value):
        cur = self.root
        while True:
            if value < cur.value:
                if cur.left is not None:
                    cur = cur.left
                else:
                    newNode = BinaryTreeNode(value)
                    cur.left = newNode
                    newNode.parent = cur
                    break
            elif value > cur.value:
                if cur.right is not None:
                    cur = cur.right
                else:
                    newNode = BinaryTreeNode(value)
                    cur.right = newNode
                    newNode.parent = cur
                    break
            else:
                cur.count += 1
                break
            
    def find(self, value):
        cur = self.root
        while True:
            if cur is None:
                return None
            if value < cur.value:
                cur = cur.left
            elif value > cur.value:
                cur = cur.right
            else:
                return cur

    def L_rotate(self, node):
        p = node.parent
        l = node.left 
        if l is None:
            return
        if l.right is None:
            l.right = node
            l.parent = p
            if p is None:
                self.root = l
            node.parent = l
            node.left = None

    def R_rotate(self, node):
        p = node.parent
        r = node.right
        if r is None:
            return
        if r.left is None:
            r.left = node
            r.parent = p
            if p is None:
                self.root = r
            node.parent = r
            node.right = None
        
    def pullup(self, node, cur):
        if cur.parent is None:
            node.parent = None
            self.root = node
        else:
            if cur.value > cur.parent.value:
                cur.parent.right = node
            else:
                cur.parent.left = node
            if node is not None:
                node.parent = cur.parent

    def delete(self, value):
        cur = self.find(value)
        if cur is None:
            return
        cur.count -= 1
        if cur.count == 0:
            if cur.left is None:
                self.pullup(cur.right, cur)
                return
            if cur.right is None:
                self.pullup(cur.left, cur)
                return 
            A = cur.right.height
            B = cur.left.height
            if A > B:
                self.pullup(cur.right, cur)
                r = cur.right
                l = cur.left
                while r.left is not None:
                    r = r.left
                r.left = l
                l.parent = r
            else:
                self.pullup(cur.left, cur)
                r = cur.right
                l = cur.left
                while l.right is not None:
                    l = l.right
                l.right = r
                r.parent = l

if __name__ == '__main__':
    tree = BinaryTree(0)
    a = np.random.randint(0, 2000, (100000,))
    for i in tqdm(range(len(a))):
        out = tree.find(a[i])
        if out is None:
            tree.insert(a[i])
        else:
            tree.delete(a[i])
    print(tree.root.height, np.log(tree.size) / np.log(2))
