var APP_WIDTH = 600;
var APP_HEIGHT = 350;
var GRAPH_SIZE_PRIM = 5;

var app_mst_prim = new PIXI.Application( APP_WIDTH, 
                                         APP_HEIGHT, 
                                         { antialias: true } );

var _divContainer_mst_prim = document.getElementById( "container2d_graphs_mst_prim" );
_divContainer_mst_prim.appendChild( app_mst_prim.view );

var _graph_mst_prim = new LGraph();
_graph_mst_prim.isWeighted = true;

// create a simple graph
// create the nodes
for ( var q = 0; q < GRAPH_SIZE_PRIM; q++ )
{
	var x = 0;
	var y = 0;
	if ( q > 0 )
	{
		var l = 100 + Math.random() * ( 50 );
		var t = ( q - 1 ) * ( Math.PI / 2 ) + Math.random() * ( Math.PI / 2 );

		x = l * Math.cos( t );
		y = l * Math.sin( t );
	}
	var _node = _graph_mst_prim.insertNode( q + 1, q );
	_node.vContainer.x = x;
	_node.vContainer.y = y;
}
// create the edges
var _nodes = _graph_mst_prim.nodes();
var _edges = null;
for ( var q = 1; q <= 4; q++ )
{
	var _dx = _nodes[0].vContainer.x - _nodes[q].vContainer.x;
	var _dy = _nodes[0].vContainer.y - _nodes[q].vContainer.y;
	var _d = Math.sqrt( _dx * _dx + _dy * _dy );

	_edges = _graph_mst_prim.insertEdge( _nodes[0], _nodes[q], _d );
	if ( _edges )
	{
		_edges[0].graphics_setEndPoints( _nodes[0].vContainer.x, _nodes[0].vContainer.y,
										 _nodes[q].vContainer.x, _nodes[q].vContainer.y );
	}
}

for ( var q = 1; q <= 4; q++ )
{
	var _nq = ( q == 4 ) ? ( 1 ) : ( q + 1 );
	var _dx = _nodes[_nq].vContainer.x - _nodes[q].vContainer.x;
	var _dy = _nodes[_nq].vContainer.y - _nodes[q].vContainer.y;
	var _d = Math.sqrt( _dx * _dx + _dy * _dy );

	_edges = _graph_mst_prim.insertEdge( _nodes[q], _nodes[_nq], _d );
	if ( _edges )
	{
		_edges[0].graphics_setEndPoints( _nodes[q].vContainer.x, _nodes[q].vContainer.y,
										 _nodes[_nq].vContainer.x, _nodes[_nq].vContainer.y );
	}
}

app_mst_prim.stage.addChild( _graph_mst_prim.vContainer );
_graph_mst_prim.vContainer.x = APP_WIDTH / 2;
_graph_mst_prim.vContainer.y = APP_HEIGHT / 2;


function onCalcSpanningTree_prim()
{
	if ( _graph_mst_prim )
	{
		var _mst = _graph_mst_prim.mst_prim();
		//_graph_mst_prim.test_priorityQueue();
	}
}

function onResetGraphics()
{
	if ( _graph_mst_prim )
	{
		_graph_mst_prim.resetGraphics();
	}
}

var _inputButton_mst_prim = document.createElement( 'INPUT' );
_inputButton_mst_prim.setAttribute( 'type', 'button' );
_inputButton_mst_prim.setAttribute( 'value', 'MST_prim' );
_inputButton_mst_prim.onclick = onCalcSpanningTree_prim;
_divContainer_mst_prim.appendChild( _inputButton_mst_prim );


var _inputButton_normal = document.createElement( 'INPUT' );
_inputButton_normal.setAttribute( 'type', 'button' );
_inputButton_normal.setAttribute( 'value', 'reset' );
_inputButton_normal.onclick = onResetGraphics;
_divContainer_mst_prim.appendChild( _inputButton_normal );

