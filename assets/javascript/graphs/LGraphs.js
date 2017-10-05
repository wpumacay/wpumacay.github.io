/// <reference path="../pixi.js.d.ts"/>
var LGraphEdge = /** @class */ (function () {
    function LGraphEdge(data, fromNode, toNode) {
        this.m_data = data;
        this.m_nodeFrom = fromNode;
        this.m_nodeTo = toNode;
    }
    LGraphEdge.prototype.from = function () {
        return this.m_nodeFrom;
    };
    LGraphEdge.prototype.to = function () {
        return this.m_nodeTo;
    };
    return LGraphEdge;
}());
var LGraphNode = /** @class */ (function () {
    function LGraphNode(data, id) {
        this.m_data = data;
        this.id = id;
        this.vContainer = new PIXI.Container();
        this.vGraphics = new PIXI.Graphics();
        this.vGraphics.lineStyle(1, 0x0000ff, 1);
        this.vGraphics.beginFill(0x0000ff, 0.75);
        this.vGraphics.drawCircle(0, 0, 10);
        this.vContainer.addChild(this.vGraphics);
    }
    LGraphNode.prototype.getData = function () {
        return this.m_data;
    };
    LGraphNode.prototype.getEdges = function () {
        return this.m_edges;
    };
    LGraphNode.prototype.addEdge = function (edge) {
        this.m_edges.push(edge);
    };
    return LGraphNode;
}());
var LGraph = /** @class */ (function () {
    function LGraph() {
        this.m_nodes = new Array();
        this.vContainer = new PIXI.Container();
        this.m_nNodes = 0;
    }
    LGraph.prototype.insertNode = function (nData, nId) {
        var _node = new LGraphNode(nData, nId);
        this.m_nodes.push(_node);
        this.vContainer.addChild(_node.vContainer);
        this.m_nNodes++;
        return _node;
    };
    LGraph.prototype.insertEdge = function (nFrom, nTo, eData) {
        var _edge_direct = new LGraphEdge(eData, nFrom, nTo);
        nFrom.addEdge(_edge_direct);
        var _edge_reverse = new LGraphEdge(eData, nTo, nFrom);
        nTo.addEdge(_edge_reverse);
    };
    return LGraph;
}());
