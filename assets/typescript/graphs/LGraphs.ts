
/// <reference path="../pixi.js.d.ts"/>

class LGraphEdge
{

	private m_data 		: number;
	private m_nodeFrom 	: LGraphNode;
	private m_nodeTo 	: LGraphNode;

	constructor( data : number, 
				 fromNode : LGraphNode, 
				 toNode : LGraphNode )
	{
		this.m_data 	= data;
		this.m_nodeFrom = fromNode;
		this.m_nodeTo 	= toNode;
	}

	public from() : LGraphNode
	{
		return this.m_nodeFrom;
	}

	public to() : LGraphNode
	{
		return this.m_nodeTo;
	}
}


class LGraphNode
{
	// public **********
	public id 		  : number;
	public vContainer : PIXI.Container;
	public vGraphics  : PIXI.Graphics;

	// private **********
	private m_data : number;
	private m_edges : Array<LGraphEdge>;

	constructor( data: number, id : number ) 
	{
		this.m_data = data;
		this.id = id;

		this.vContainer = new PIXI.Container();
		this.vGraphics = new PIXI.Graphics();

		this.vGraphics.lineStyle( 1, 0x0000ff, 1 );
		this.vGraphics.beginFill( 0x0000ff, 0.75 );
		this.vGraphics.drawCircle( 0, 0, 10 );

		this.vContainer.addChild( this.vGraphics );
	}

	public getData() : number
	{
		return this.m_data;
	}

	public getEdges() : Array<LGraphEdge>
	{
		return this.m_edges;
	}

	public addEdge( edge: LGraphEdge ) : void
	{
		this.m_edges.push( edge );
	}
}


class LGraph
{
	public vContainer : PIXI.Container;
	
	private m_nNodes : number;
	private m_nodes : Array<LGraphNode>;

	constructor() 
	{
		this.m_nodes = new Array<LGraphNode>();
		this.vContainer = new PIXI.Container();
		this.m_nNodes = 0;
	}

	public insertNode( nData : number, nId : number ) : LGraphNode
	{
		let _node : LGraphNode = new LGraphNode( nData, nId );

		this.m_nodes.push( _node );
		this.vContainer.addChild( _node.vContainer );

		this.m_nNodes++;

		return _node;
	}

	public insertEdge( nFrom: LGraphNode, nTo: LGraphNode, eData: number ) : void
	{
		let _edge_direct : LGraphEdge = new LGraphEdge( eData, nFrom, nTo );
		nFrom.addEdge( _edge_direct );

		let _edge_reverse : LGraphEdge = new LGraphEdge( eData, nTo, nFrom );
		nTo.addEdge( _edge_reverse );
	}
}