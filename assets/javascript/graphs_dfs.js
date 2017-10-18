var APP_WIDTH = 600;
var APP_HEIGHT = 350;
var GRAPH_SIZE_DFS = 10;

var app_dfs = new PIXI.Application( APP_WIDTH, 
                                    APP_HEIGHT, 
                                    { antialias: true } );

app_dfs.stage.interactive = true;

var _divContainer_dfs = document.getElementById( "container2d_graphs_dfs" );
_divContainer_dfs.appendChild( app_dfs.view );

var _graph_dfs = null;

function createGraphList_dfs()
{
    if ( _graph_dfs != null )
    {
        app_dfs.stage.removeChild( _graph_dfs.vContainer );
        _graph_dfs.free();
    }

    _graph_dfs = new LGraph();

    for ( var q = 0; q < GRAPH_SIZE_DFS; q++ )
    {
        var _node = _graph_dfs.insertNode( q + 1, q );

        _node.vContainer.x = 50 + Math.random() * ( APP_WIDTH - 100 );
        _node.vContainer.y = 50 + Math.random() * ( APP_HEIGHT - 100 );
    }

    // Create a sparse graph
    var _vertices = [];
    var _nodes = _graph_dfs.nodes();
    for ( var q = 0; q < _graph_dfs.numNodes(); q++ )
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
            var _edges = _graph_dfs.insertEdge( _nodes[_indx1], _nodes[_indx2] );

            if ( _edges )
            {
                _edges[0].graphics_setEndPoints( _nodes[_indx1].vContainer.x, _nodes[_indx1].vContainer.y,
                                                 _nodes[_indx2].vContainer.x, _nodes[_indx2].vContainer.y );
                _edges[1].graphics_setEndPoints( _nodes[_indx2].vContainer.x, _nodes[_indx2].vContainer.y,
                                                 _nodes[_indx1].vContainer.x, _nodes[_indx1].vContainer.y );
            }
        }
    }

    app_dfs.stage.addChild( _graph_dfs.vContainer );
}

function dfs( startNode )
{
    _graph_dfs.dfs( startNode );
}

createGraphList_dfs();
if ( _graph_dfs )
{
    var _nodes = _graph_dfs.nodes();
    dfs( _nodes[0] );
}

function onChangeSize_dfs()
{
    GRAPH_SIZE_DFS = document.getElementById( 'graph_size_dfs' ).value;
    GRAPH_SIZE_DFS = parseInt( GRAPH_SIZE_DFS, 10 );

    createGraphList_dfs();
}

function onAnimate_dfs()
{
    if ( _graph_dfs )
    {
        var _nodes = _graph_dfs.nodes();
        dfs( _nodes[0] );
        _graph_dfs.animate_dfs_start();
    }
}
    

// Add inputs
var _inputNum = document.createElement( 'INPUT' );
_inputNum.setAttribute( 'type', 'number' );
_inputNum.setAttribute( 'value', GRAPH_SIZE_DFS );
_inputNum.setAttribute( 'id', 'graph_size_dfs' );
_divContainer_dfs.appendChild( _inputNum );

var _inputButton = document.createElement( 'INPUT' );
_inputButton.setAttribute( 'type', 'button' );
_inputButton.setAttribute( 'value', 'setSize' );
_inputButton.onclick = onChangeSize_dfs;
_divContainer_dfs.appendChild( _inputButton );

var _inputButton_run = document.createElement( 'INPUT' );
_inputButton_run.setAttribute( 'type', 'button' );
_inputButton_run.setAttribute( 'value', 'run_dfs' );
_inputButton_run.onclick = onAnimate_dfs;
_divContainer_dfs.appendChild( _inputButton_run );