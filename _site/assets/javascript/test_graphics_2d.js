var APP_WIDTH = 400;
var APP_HEIGHT = 300;

var app = new PIXI.Application( APP_WIDTH, 
                                APP_HEIGHT, 
                                { antialias: true } );

var _canvas = document.getElementById( "container2d" );
_canvas.appendChild( app.view );

var _graph = new LGraph();

for ( var q = 0; q < 10; q++ )
{
    var _node = _graph.insertNode( q + 1, q );

    _node.vContainer.x = Math.random() * APP_WIDTH;
    _node.vContainer.y = Math.random() * APP_HEIGHT;
}


app.stage.addChild( _graph.vContainer );

