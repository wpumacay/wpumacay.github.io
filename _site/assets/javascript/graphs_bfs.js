var APP_WIDTH = 600;
var APP_HEIGHT = 350;
var GRAPH_SIZE_BFS = 10;

var app_bfs = new PIXI.Application( APP_WIDTH, 
                                    APP_HEIGHT, 
                                    { antialias: true } );

app_bfs.stage.interactive = true;

var _divContainer_bfs = document.getElementById( "container2d_graphs_bfs" );
_divContainer_bfs.appendChild( app_bfs.view );

var _graph_bfs = null;

function createGraphList_bfs()
{
    if ( _graph_bfs != null )
    {
        app_bfs.stage.removeChild( _graph_bfs.vContainer );
        _graph_bfs.free();
    }

    _graph_bfs = new LGraph();

    for ( var q = 0; q < GRAPH_SIZE_BFS; q++ )
    {
        var _node = _graph_bfs.insertNode( q + 1, q );

        _node.vContainer.x = 50 + Math.random() * ( APP_WIDTH - 100 );
        _node.vContainer.y = 50 + Math.random() * ( APP_HEIGHT - 100 );
    }

    // Create a sparse graph
    var _vertices = [];
    var _nodes = _graph_bfs.nodes();
    for ( var q = 0; q < _graph_bfs.numNodes(); q++ )
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
            var _edges = _graph_bfs.insertEdge( _nodes[_indx1], _nodes[_indx2] );

            if ( _edges )
            {
                _edges[0].graphics_setEndPoints( _nodes[_indx1].vContainer.x, _nodes[_indx1].vContainer.y,
                                                 _nodes[_indx2].vContainer.x, _nodes[_indx2].vContainer.y );
                _edges[1].graphics_setEndPoints( _nodes[_indx2].vContainer.x, _nodes[_indx2].vContainer.y,
                                                 _nodes[_indx1].vContainer.x, _nodes[_indx1].vContainer.y );
            }
        }
    }

    app_bfs.stage.addChild( _graph_bfs.vContainer );
}

function bfs( startNode )
{
    _graph_bfs.bfs( startNode );
}

createGraphList_bfs();
if ( _graph_bfs )
{
    var _nodes = _graph_bfs.nodes();
    bfs( _nodes[0] );
}

function onChangeSize_bfs()
{
    GRAPH_SIZE_BFS = document.getElementById( 'graph_size_bfs' ).value;
    GRAPH_SIZE_BFS = parseInt( GRAPH_SIZE_BFS, 10 );

    createGraphList_bfs();
}

function onAnimate_bfs()
{
    if ( _graph_bfs )
    {
        var _nodes = _graph_bfs.nodes();
        bfs( _nodes[0] );
        _graph_bfs.animate_bfs_start();
    }
}


// Add inputs
var _inputNum = document.createElement( 'INPUT' );
_inputNum.setAttribute( 'type', 'number' );
_inputNum.setAttribute( 'value', GRAPH_SIZE_BFS );
_inputNum.setAttribute( 'id', 'graph_size_bfs' );
_divContainer_bfs.appendChild( _inputNum );

var _inputButton = document.createElement( 'INPUT' );
_inputButton.setAttribute( 'type', 'button' );
_inputButton.setAttribute( 'value', 'setSize' );
_inputButton.onclick = onChangeSize_bfs;
_divContainer_bfs.appendChild( _inputButton );

var _inputButton_run = document.createElement( 'INPUT' );
_inputButton_run.setAttribute( 'type', 'button' );
_inputButton_run.setAttribute( 'value', 'run_bfs' );
_inputButton_run.onclick = onAnimate_bfs;
_divContainer_bfs.appendChild( _inputButton_run );