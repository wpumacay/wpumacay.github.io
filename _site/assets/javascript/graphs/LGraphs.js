/// <reference path="LGraphs.ts"/>
var LHeap = /** @class */ (function () {
    function LHeap(arr) {
        this.m_arr = new Array();
        for (var q = 0; q < arr.length; q++) {
            this.m_arr.push(arr[q]);
        }
    }
    LHeap.prototype.extractMinimum = function () {
        if (this.m_arr.length < 1) {
            return null;
        }
        if (this.m_arr.length == 1) {
            return this.m_arr.shift();
        }
        var _min = this.m_arr[0];
        this.m_arr[0] = this.m_arr.pop();
        this.minHeapify(0);
        return _min;
    };
    LHeap.prototype.isEmpty = function () {
        return this.m_arr.length < 1;
    };
    LHeap.build_maxHeap = function (arr) {
        var _res = new LHeap(arr);
        for (var q = Math.ceil(arr.length / 2); q >= 1; q--) {
            _res.maxHeapify(q);
        }
        return _res;
    };
    LHeap.build_minHeap = function (arr) {
        var _res = new LHeap(arr);
        for (var q = Math.ceil(arr.length / 2); q >= 1; q--) {
            _res.minHeapify(q);
        }
        return _res;
    };
    LHeap.prototype.maxHeapify = function (i) {
        var _l = this.left(i);
        var _r = this.right(i);
        var _largest = i;
        if (_l < this.m_arr.length &&
            this.m_arr[_l].priority < this.m_arr[i].priority) {
            _largest = _l;
        }
        else {
            _largest = i;
        }
        if (_r < this.m_arr.length &&
            this.m_arr[_r].priority < this.m_arr[_largest].priority) {
            _largest = _r;
        }
        if (_largest != i) {
            this.swap(i, _largest);
            this.maxHeapify(_largest);
        }
    };
    LHeap.prototype.minHeapify = function (i) {
        var _l = this.left(i);
        var _r = this.right(i);
        var _smallest = i;
        if (_l < this.m_arr.length &&
            this.m_arr[_l].priority < this.m_arr[_smallest].priority) {
            _smallest = _l;
        }
        if (_r < this.m_arr.length &&
            this.m_arr[_r].priority < this.m_arr[_smallest].priority) {
            _smallest = _r;
        }
        if (_smallest != i) {
            this.swap(i, _smallest);
            this.maxHeapify(_smallest);
        }
    };
    LHeap.prototype.left = function (i) {
        return 2 * i + 1;
    };
    LHeap.prototype.right = function (i) {
        return 2 * i + 2;
    };
    LHeap.prototype.parent = function (i) {
        return Math.floor((i - 1) / 2);
    };
    LHeap.prototype.swap = function (i, j) {
        var _tmp = this.m_arr[i];
        this.m_arr[i] = this.m_arr[j];
        this.m_arr[j] = _tmp;
    };
    return LHeap;
}());
var LPriorityQueue = /** @class */ (function () {
    function LPriorityQueue(nodes) {
        this.m_heap = LHeap.build_minHeap(nodes);
    }
    LPriorityQueue.prototype.extractMin = function () {
        var _res = this.m_heap.extractMinimum();
        return _res;
    };
    LPriorityQueue.prototype.isEmpty = function () {
        return this.m_heap.isEmpty();
    };
    return LPriorityQueue;
}());
/// <reference path="../pixi.js.d.ts"/>
/// <reference path="LPriorityQueue.ts"/>
var LGraphEdge = /** @class */ (function () {
    function LGraphEdge(data, fromNode, toNode) {
        this.vContainer = new PIXI.Container();
        this.vGraphics = new PIXI.Graphics();
        this.m_p1x = 0;
        this.m_p1y = 0;
        this.m_p2x = 0;
        this.m_p2y = 0;
        this.color = 0x0000ff;
        this.vGraphics.beginFill(0x0000ff);
        this.vGraphics.lineStyle(1, 0x0000ff, 1);
        this.vContainer.addChild(this.vGraphics);
        this.m_data = data;
        this.m_nodeFrom = fromNode;
        this.m_nodeTo = toNode;
        this.isWeighted = false;
        var _style = new PIXI.TextStyle({
            fontFamily: 'Arial',
            fontSize: 10,
            fill: '#ffffff'
        });
        this.vText = new PIXI.Text('' + this.m_data.toFixed(2), _style);
        this.vText.anchor.set(0.5);
    }
    LGraphEdge.prototype.free = function () {
        this.vContainer.removeChild(this.vGraphics);
        this.vContainer.parent.removeChild(this.vContainer);
        this.vContainer = null;
        this.vGraphics = null;
    };
    LGraphEdge.prototype.from = function () {
        return this.m_nodeFrom;
    };
    LGraphEdge.prototype.to = function () {
        return this.m_nodeTo;
    };
    LGraphEdge.prototype.graphics_setEndPoints = function (p1x, p1y, p2x, p2y) {
        this.m_p1x = p1x;
        this.m_p1y = p1y;
        this.m_p2x = p2x;
        this.m_p2y = p2y;
        this.vGraphics.clear();
        this.vGraphics.beginFill(0x0000ff);
        this.vGraphics.lineStyle(1, 0x0000ff, 1);
        this.vGraphics.moveTo(this.m_p1x, this.m_p1y);
        this.vGraphics.lineTo(this.m_p2x, this.m_p2y);
        if (this.isWeighted) {
            this.vContainer.addChild(this.vText);
            this.vText.x = 0.5 * (this.m_p1x + this.m_p2x);
            this.vText.y = 0.5 * (this.m_p1y + this.m_p2y);
        }
    };
    LGraphEdge.prototype.graphics_redraw = function () {
        this.vGraphics.clear();
        this.vGraphics.beginFill(this.color);
        this.vGraphics.lineStyle(1, this.color, 1);
        this.vGraphics.moveTo(this.m_p1x, this.m_p1y);
        this.vGraphics.lineTo(this.m_p2x, this.m_p2y);
    };
    return LGraphEdge;
}());
var LGraphNode = /** @class */ (function () {
    function LGraphNode(data, id) {
        this.priority = -1;
        this.level = -1;
        this.parent = null;
        this.color = 0xffffff;
        this.m_data = data;
        this.m_edges = new Array();
        this.id = id;
        this.vContainer = new PIXI.Container();
        this.vGraphics = new PIXI.Graphics();
        var _style = new PIXI.TextStyle({
            fontFamily: 'Arial',
            fontSize: 10
        });
        this.vText = new PIXI.Text('' + this.id, _style);
        this.vText.anchor.set(0.5);
        this.vGraphics.lineStyle(1, 0x0000ff, 1);
        this.vGraphics.beginFill(this.color, 0.75);
        this.vGraphics.drawCircle(0, 0, 10);
        this.vContainer.addChild(this.vGraphics);
        this.vContainer.addChild(this.vText);
    }
    LGraphNode.prototype.free = function () {
        for (var q = 0; q < this.m_edges.length; q++) {
            this.m_edges[q].free();
            this.m_edges[q] = null;
        }
        this.vContainer.removeChild(this.vGraphics);
        this.vContainer.removeChild(this.vText);
        this.vGraphics = null;
        this.vText = null;
    };
    LGraphNode.prototype.getData = function () {
        return this.m_data;
    };
    LGraphNode.prototype.getEdges = function () {
        return this.m_edges;
    };
    LGraphNode.prototype.addEdge = function (edge) {
        this.m_edges.push(edge);
    };
    LGraphNode.prototype.graphics_redraw = function () {
        this.vGraphics.clear();
        this.vGraphics.lineStyle(1, 0x0000ff, 1);
        this.vGraphics.beginFill(this.color, 1);
        this.vGraphics.drawCircle(0, 0, 10);
    };
    return LGraphNode;
}());
var LGraph = /** @class */ (function () {
    function LGraph() {
        this.m_nodes = new Array();
        this.vContainer = new PIXI.Container();
        this.vBackContainer = new PIXI.Container();
        this.vFrontContainer = new PIXI.Container();
        this.vContainer.addChild(this.vBackContainer);
        this.vContainer.addChild(this.vFrontContainer);
        this.visitCounter = 0;
        this.visitBuff_dfs = new Array();
        this.visitBuff_bfs = new Array();
        this.m_nNodes = 0;
        this.bfs_st_nodes = new Array();
        this.bfs_st_edges = new Array();
        this.dfs_st_nodes = new Array();
        this.dfs_st_edges = new Array();
        this.isWeighted = false;
        this.isDirected = false;
    }
    LGraph.prototype.free = function () {
        for (var q = 0; q < this.m_nNodes; q++) {
            this.m_nodes[q].free();
            this.m_nodes[q] = null;
        }
        this.visitBuff_bfs = null;
        this.visitBuff_dfs = null;
        this.vContainer.removeChild(this.vBackContainer);
        this.vContainer.removeChild(this.vFrontContainer);
        this.vBackContainer = null;
        this.vFrontContainer = null;
    };
    LGraph.prototype.insertNode = function (nData, nId) {
        var _node = new LGraphNode(nData, nId);
        this.m_nodes.push(_node);
        this.m_nNodes++;
        // Graphics part
        this.vFrontContainer.addChild(_node.vContainer);
        return _node;
    };
    LGraph.prototype.numNodes = function () {
        return this.m_nNodes;
    };
    LGraph.prototype.nodes = function () {
        return this.m_nodes;
    };
    LGraph.prototype.insertEdge = function (nFrom, nTo, eData) {
        var _edges_from = nFrom.getEdges();
        for (var q = 0; q < _edges_from.length; q++) {
            if ((_edges_from[q].to()).id == nTo.id) {
                return;
            }
        }
        var _edge_direct = new LGraphEdge(eData, nFrom, nTo);
        nFrom.addEdge(_edge_direct);
        _edge_direct.isWeighted = this.isWeighted;
        var _edge_reverse = new LGraphEdge(eData, nTo, nFrom);
        nTo.addEdge(_edge_reverse);
        _edge_reverse.isWeighted = this.isWeighted;
        // Graphics part
        this.vBackContainer.addChild(_edge_direct.vContainer);
        //this.vBackContainer.addChild( _edge_reverse.vContainer ); // the reverse is just kept for safe delete
        return [_edge_direct, _edge_reverse];
    };
    LGraph.prototype.resetGraphics = function () {
        for (var q = 0; q < this.m_nodes.length; q++) {
            this.m_nodes[q].color = 0xffffff;
            this.m_nodes[q].graphics_redraw();
            var _edges = this.m_nodes[q].getEdges();
            for (var p = 0; p < _edges.length; p++) {
                _edges[p].color = 0x0000ff;
                _edges[p].graphics_redraw();
            }
        }
    };
    LGraph.prototype.test_priorityQueue = function () {
        for (var q = 0; q < this.m_nodes.length; q++) {
            this.m_nodes[q].priority = (this.m_nodes[q].id + 1) * 10;
        }
        var _pq = new LPriorityQueue(this.m_nodes);
        console.log('test pq');
        while (_pq.isEmpty() == false) {
            var _minNode = _pq.extractMin();
            console.log('p: ' + _minNode.priority);
        }
    };
    LGraph.prototype.mst_prim = function () {
        var _mst = new LGraph();
        _mst.isWeighted = true;
        return _mst;
    };
    LGraph.prototype.spanningTree_bfs = function () {
        this.resetGraphics();
        // just pick the first node as start
        this.bfs(this.m_nodes[0]);
        // show bfs-spanning tree in the graph
        for (var q = 0; q < this.bfs_st_nodes.length; q++) {
            this.bfs_st_nodes[q].color = 0x00ff00;
            this.bfs_st_nodes[q].graphics_redraw();
        }
        for (var q = 0; q < this.bfs_st_edges.length; q++) {
            this.bfs_st_edges[q].color = 0x00ff00;
            this.bfs_st_edges[q].graphics_redraw();
        }
    };
    LGraph.prototype.spanningTree_dfs = function () {
        this.resetGraphics();
        // just pick the first node as start
        this.dfs();
        // show bfs-spanning tree in the graph
        for (var q = 0; q < this.dfs_st_nodes.length; q++) {
            this.dfs_st_nodes[q].color = 0x00ff00;
            this.dfs_st_nodes[q].graphics_redraw();
        }
        for (var q = 0; q < this.dfs_st_edges.length; q++) {
            this.dfs_st_edges[q].color = 0x00ff00;
            this.dfs_st_edges[q].graphics_redraw();
        }
    };
    LGraph.prototype.bfs = function (startNode) {
        for (var q = 0; q < this.m_nNodes; q++) {
            this.m_nodes[q].level = -1;
            this.m_nodes[q].parent = null;
        }
        // initialize animation stuff
        this.visitBuff_bfs = new Array();
        this.bfs_st_nodes = new Array();
        this.bfs_st_edges = new Array();
        var _frontier = new Array();
        var _iLevel = 1;
        _frontier.push(startNode);
        startNode.level = 0;
        this.visitBuff_bfs.push(startNode);
        this.bfs_st_nodes.push(startNode);
        while (_frontier.length != 0) {
            var _next = new Array();
            var _color = Math.floor(0xffffff * Math.random());
            for (var q = 0; q < _frontier.length; q++) {
                var _u = _frontier[q];
                var _edges = _u.getEdges();
                for (var e = 0; e < _edges.length; e++) {
                    var _v = _edges[e].to();
                    if (_v.level == -1) {
                        _v.level = _iLevel;
                        _v.parent = _u;
                        _next.push(_v);
                        // Graphics stuff
                        _v.color = _color;
                        _v.graphics_redraw();
                        // animation stuff
                        this.visitBuff_bfs.push(_v);
                        // spanning tree stuff
                        this.bfs_st_nodes.push(_v);
                        this.bfs_st_edges.push(_edges[e]);
                    }
                }
            }
            _frontier = _next;
            _iLevel++;
        }
    };
    LGraph.prototype.animate_bfs_start = function () {
        for (var q = 0; q < this.visitBuff_bfs.length; q++) {
            this.visitBuff_bfs[q].vContainer.visible = false;
        }
        this.visitCounter = 0;
        this.visitTimeout = setInterval(this.animate_bfs_step, 1000, this);
    };
    LGraph.prototype.animate_bfs_step = function (self) {
        self.visitBuff_bfs[self.visitCounter].vContainer.visible = true;
        self.visitCounter++;
        if (self.visitCounter == self.visitBuff_bfs.length) {
            self.animate_bfs_stop();
        }
    };
    LGraph.prototype.animate_bfs_stop = function () {
        this.visitCounter = 0;
        clearInterval(this.visitTimeout);
    };
    LGraph.prototype.dfs = function () {
        // clean animation stuff
        this.visitBuff_dfs = new Array();
        this.dfs_st_nodes = new Array();
        this.dfs_st_edges = new Array();
        // Initialize dfs stuff for all nodes
        for (var q = 0; q < this.m_nodes.length; q++) {
            this.m_nodes[q].parent = null;
        }
        for (var q = 0; q < this.m_nodes.length; q++) {
            if (this.m_nodes[q].parent == null) {
                this.m_nodes[q].parent = this.m_nodes[q];
                this.visitBuff_dfs.push(this.m_nodes[q]);
                this.dfs_st_nodes.push(this.m_nodes[q]);
                this.dfs_visit(this.m_nodes[q]);
            }
        }
    };
    LGraph.prototype.dfs_visit = function (u) {
        var _edges = u.getEdges();
        for (var q = 0; q < _edges.length; q++) {
            var _v = _edges[q].to();
            if (_v.parent == null) {
                _v.parent = u;
                // animation stuff
                this.visitBuff_dfs.push(_v);
                this.dfs_st_nodes.push(_v);
                this.dfs_st_edges.push(_edges[q]);
                this.dfs_visit(_v);
            }
        }
    };
    LGraph.prototype.animate_dfs_start = function () {
        for (var q = 0; q < this.visitBuff_dfs.length; q++) {
            this.visitBuff_dfs[q].vContainer.visible = false;
        }
        this.visitCounter = 0;
        this.visitTimeout = setInterval(this.animate_dfs_step, 1000, this);
    };
    LGraph.prototype.animate_dfs_step = function (self) {
        self.visitBuff_dfs[self.visitCounter].vContainer.visible = true;
        self.visitCounter++;
        if (self.visitCounter == self.visitBuff_dfs.length) {
            self.animate_dfs_stop();
        }
    };
    LGraph.prototype.animate_dfs_stop = function () {
        this.visitCounter = 0;
        clearInterval(this.visitTimeout);
    };
    // Minimum spanning trees code
    LGraph.prototype.minimumSpanningTree_prim = function () {
        var _mst = new LGraph();
        return _mst;
    };
    return LGraph;
}());
var LGraphMat = /** @class */ (function () {
    function LGraphMat(nNodes) {
        this.m_nNodes = nNodes;
        this.m_mat = new Array();
        for (var q = 0; q < this.m_nNodes; q++) {
            var _row = new Array();
            for (var p = 0; p < this.m_nNodes; p++) {
                _row.push(0);
            }
            this.m_mat.push(_row);
        }
        this.vNodes = new Array();
        for (var q = 0; q < this.m_nNodes; q++) {
            this.vNodes.push(new PIXI.Point());
        }
        this.vContainer = new PIXI.Container();
        this.vGraphics = new PIXI.Graphics();
        this.vContainer.addChild(this.vGraphics);
    }
    LGraphMat.prototype.free = function () {
        this.vContainer.removeChild(this.vGraphics);
        this.vGraphics.clear();
        this.vGraphics = null;
        this.vNodes = null;
        this.m_mat = null;
    };
    LGraphMat.prototype.insertEdge = function (id1, id2, w) {
        if (id1 < 0 || id1 > this.m_nNodes - 1) {
            return;
        }
        if (id2 < 0 || id2 > this.m_nNodes - 1) {
            return;
        }
        this.m_mat[id1][id2] = w;
        this.m_mat[id2][id1] = w;
    };
    LGraphMat.prototype.graphics_redrawGraph = function () {
        this.vGraphics.clear();
        for (var q = 0; q < this.m_nNodes; q++) {
            for (var p = q + 1; p < this.m_nNodes; p++) {
                if (this.m_mat[q][p] != 0) {
                    this.vGraphics.lineStyle(1, 0x0000ff, 1);
                    this.vGraphics.beginFill(0xffffff, 0.75);
                    this.vGraphics.drawCircle(this.vNodes[q].x, this.vNodes[q].y, 10);
                    this.vGraphics.drawCircle(this.vNodes[p].x, this.vNodes[p].y, 10);
                    this.vGraphics.beginFill(0x0000ff);
                    this.vGraphics.lineStyle(1, 0x0000ff, 1);
                    this.vGraphics.moveTo(this.vNodes[q].x, this.vNodes[q].y);
                    this.vGraphics.lineTo(this.vNodes[p].x, this.vNodes[p].y);
                }
            }
        }
    };
    return LGraphMat;
}());
