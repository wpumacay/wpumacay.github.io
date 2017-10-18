var APP_WIDTH = 600;
var APP_HEIGHT = 350;

var app = new PIXI.Application( APP_WIDTH, 
                                APP_HEIGHT, 
                                { antialias: true } );

app.stage.interactive = true;

var _canvas = document.getElementById( "container2d" );
_canvas.appendChild( app.view );

var _graph = new LGraph();

for ( var q = 0; q < 10; q++ )
{
    var _node = _graph.insertNode( q + 1, q );

    _node.vContainer.x = 50 + Math.random() * ( APP_WIDTH - 100 );
    _node.vContainer.y = 50 + Math.random() * ( APP_HEIGHT - 100 );
}

// Create a dense graph
var _nodes = _graph.nodes();
for ( var q = 0; q < _graph.numNodes(); q++ )
{
    for ( var p = q + 1; p < _graph.numNodes(); p++ )
    {
        // create an edge
        var _edges = _graph.insertEdge( _nodes[q], _nodes[p] );
        _edges[0].graphics_setEndPoints( _nodes[q].vContainer.x, _nodes[q].vContainer.y,
                                         _nodes[p].vContainer.x, _nodes[p].vContainer.y );
        _edges[1].graphics_setEndPoints( _nodes[p].vContainer.x, _nodes[p].vContainer.y,
                                         _nodes[q].vContainer.x, _nodes[q].vContainer.y );
    }
}


app.stage.addChild( _graph.vContainer );

