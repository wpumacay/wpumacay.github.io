var APP_WIDTH = 600;
var APP_HEIGHT = 350;
var GRAPH_SIZE_ADJ_LIST = 10;

var app_adjlist = new PIXI.Application( APP_WIDTH, 
                                        APP_HEIGHT, 
                                        { antialias: true } );

app_adjlist.stage.interactive = true;

var _divContainer_adjlist = document.getElementById( "container2d_graphs_adj_list" );
_divContainer_adjlist.appendChild( app_adjlist.view );

var _graph_adjlist = null;

function createGraphList()
{
    if ( _graph_adjlist != null )
    {
        app_adjlist.stage.removeChild( _graph_adjlist.vContainer );
        _graph_adjlist.free();
    }

    _graph_adjlist = new LGraph();

    for ( var q = 0; q < GRAPH_SIZE_ADJ_LIST; q++ )
    {
        var _node = _graph_adjlist.insertNode( q + 1, q );

        _node.vContainer.x = 50 + Math.random() * ( APP_WIDTH - 100 );
        _node.vContainer.y = 50 + Math.random() * ( APP_HEIGHT - 100 );
    }

    // Create a sparse graph
    var _vertices = [];
    var _nodes = _graph_adjlist.nodes();
    for ( var q = 0; q < _graph_adjlist.numNodes(); q++ )
    {
        _vertices.push( [_nodes[q].vContainer.x, _nodes[q].vContainer.y ] );
    }

    var _indices = Delaunay.triangulate( _vertices );
    var _nTri = Math.floor( _indices.length / 3 );

    for ( var q = 0; q < _nTri; q++ )
    {
        for ( var p = 0; p < 3; p++ )
        {
            var _indx1 = _indices[3 * q + p];
            var _indx2 = _indices[3 * q + ( p + 1 ) % 3];
            var _edges = _graph_adjlist.insertEdge( _nodes[_indx1], _nodes[_indx2] );

            if ( _edges )
            {
                _edges[0].graphics_setEndPoints( _nodes[_indx1].vContainer.x, _nodes[_indx1].vContainer.y,
                                                 _nodes[_indx2].vContainer.x, _nodes[_indx2].vContainer.y );
                _edges[1].graphics_setEndPoints( _nodes[_indx2].vContainer.x, _nodes[_indx2].vContainer.y,
                                                 _nodes[_indx1].vContainer.x, _nodes[_indx1].vContainer.y );
            }
        }
    }

    app_adjlist.stage.addChild( _graph_adjlist.vContainer );
}

createGraphList();

function onChangeSize_adjList()
{
    GRAPH_SIZE_ADJ_LIST = document.getElementById( 'graph_size_list' ).value;
    GRAPH_SIZE_ADJ_LIST = parseInt( GRAPH_SIZE_ADJ_LIST, 10 );

    createGraphList();
}

// Add inputs
var _inputNum = document.createElement( 'INPUT' );
_inputNum.setAttribute( 'type', 'number' );
_inputNum.setAttribute( 'value', GRAPH_SIZE_ADJ_LIST );
_inputNum.setAttribute( 'id', 'graph_size_list' );
_divContainer_adjlist.appendChild( _inputNum );

var _inputButton = document.createElement( 'INPUT' );
_inputButton.setAttribute( 'type', 'button' );
_inputButton.setAttribute( 'value', 'setSize' );
_inputButton.onclick = onChangeSize_adjList;
_divContainer_adjlist.appendChild( _inputButton );
