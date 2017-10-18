var APP_WIDTH = 600;
var APP_HEIGHT = 350;
var GRAPH_SIZE_ADJ_MAT = 10;

var app_adjmat = new PIXI.Application( APP_WIDTH, 
                                       APP_HEIGHT, 
                                       { antialias: true } );

app_adjmat.stage.interactive = true;

var _divContainer_adjmat = document.getElementById( "container2d_graphs_adj_mat" );
_divContainer_adjmat.appendChild( app_adjmat.view );

var _graph_adjmat = null;

function createGraphMat()
{
    if ( _graph_adjmat != null )
    {
        app_adjmat.stage.removeChild( _graph_adjmat.vContainer );
        _graph_adjmat.free();
    }

    _graph_adjmat = new LGraphMat( GRAPH_SIZE_ADJ_MAT );

    for ( var q = 0; q < GRAPH_SIZE_ADJ_MAT; q++ )
    {
        var _nodePoint = _graph_adjmat.vNodes[q];

        _nodePoint.x = 50 + Math.random() * ( APP_WIDTH - 100 );
        _nodePoint.y = 50 + Math.random() * ( APP_HEIGHT - 100 );
    }

    for ( var q = 0; q < GRAPH_SIZE_ADJ_MAT; q++ )
    {
        for ( var p = q + 1; p < GRAPH_SIZE_ADJ_MAT; p++ )
        {
            _graph_adjmat.insertEdge( q, p );
        }
    }

    _graph_adjmat.graphics_redrawGraph();

    app_adjmat.stage.addChild( _graph_adjmat.vContainer );
    
}

createGraphMat();

function onChangeSize_adjMat()
{
    GRAPH_SIZE_ADJ_MAT = document.getElementById( 'graph_size_mat' ).value;
    GRAPH_SIZE_ADJ_MAT = parseInt( GRAPH_SIZE_ADJ_MAT, 10 );

    createGraphMat();
}

// Add inputs
var _inputNum = document.createElement( 'INPUT' );
_inputNum.setAttribute( 'type', 'number' );
_inputNum.setAttribute( 'value', GRAPH_SIZE_ADJ_MAT );
_inputNum.setAttribute( 'id', 'graph_size_mat' );
_divContainer_adjmat.appendChild( _inputNum );

var _inputButton = document.createElement( 'INPUT' );
_inputButton.setAttribute( 'type', 'button' );
_inputButton.setAttribute( 'value', 'setSize' );
_inputButton.onclick = onChangeSize_adjMat;
_divContainer_adjmat.appendChild( _inputButton );
