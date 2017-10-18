
/// <reference path="../pixi.js.d.ts"/>
/// <reference path="LPriorityQueue.ts"/>

class LGraphEdge
{
	// public **********
	public vContainer : PIXI.Container;
	public vGraphics  : PIXI.Graphics;
	public vText	  : PIXI.Text;
	public color 	  : number;

	public isWeighted : boolean;

	// private **********

	private m_data 		: number;
	private m_nodeFrom 	: LGraphNode;
	private m_nodeTo 	: LGraphNode;

	private m_p1x : number;
	private m_p1y : number;
	private m_p2x : number;
	private m_p2y : number;

	constructor( data : number, 
				 fromNode : LGraphNode, 
				 toNode : LGraphNode )
	{
		this.vContainer = new PIXI.Container();
		this.vGraphics = new PIXI.Graphics();
		this.m_p1x = 0;
		this.m_p1y = 0;
		this.m_p2x = 0;
		this.m_p2y = 0;		

		this.color = 0x0000ff;

		this.vGraphics.beginFill( 0x0000ff );
		this.vGraphics.lineStyle( 1, 0x0000ff, 1 );
		this.vContainer.addChild( this.vGraphics );

		this.m_data 	= data;
		this.m_nodeFrom = fromNode;
		this.m_nodeTo 	= toNode;
		this.isWeighted = false;

		let _style : PIXI.TextStyle = new PIXI.TextStyle(
											{
												fontFamily : 'Arial',
												fontSize : 10,
												fill : '#ffffff'
											} );
		this.vText = new PIXI.Text( '' + this.m_data.toFixed( 2 ), _style );
		this.vText.anchor.set( 0.5 );
	}

	public data() : number
	{
		return this.m_data;
	}

	public free() : void
	{
		this.vContainer.removeChild( this.vGraphics );
		this.vContainer.parent.removeChild( this.vContainer );

		this.vContainer = null;
		this.vGraphics = null;
	}

	public from() : LGraphNode
	{
		return this.m_nodeFrom;
	}

	public to() : LGraphNode
	{
		return this.m_nodeTo;
	}

	public graphics_setEndPoints( p1x : number, p1y : number,
								  p2x : number, p2y : number )
	{
		this.m_p1x = p1x;
		this.m_p1y = p1y;
		this.m_p2x = p2x;
		this.m_p2y = p2y;

		this.vGraphics.clear();
		this.vGraphics.beginFill( 0x0000ff );
		this.vGraphics.lineStyle( 1, 0x0000ff, 1 );
		this.vGraphics.moveTo( this.m_p1x, this.m_p1y );
		this.vGraphics.lineTo( this.m_p2x, this.m_p2y );

		if ( this.isWeighted )
		{
			this.vContainer.addChild( this.vText );
			this.vText.x = 0.5 * ( this.m_p1x + this.m_p2x );
			this.vText.y = 0.5 * ( this.m_p1y + this.m_p2y );
		}
	}

	public graphics_redraw() : void
	{
		this.vGraphics.clear();
		this.vGraphics.beginFill( this.color );
		this.vGraphics.lineStyle( 1, this.color, 1 );
		this.vGraphics.moveTo( this.m_p1x, this.m_p1y );
		this.vGraphics.lineTo( this.m_p2x, this.m_p2y );
	}
}


class LGraphNode
{
	// public **********
	public id 		  : number;
	public level 	  : number;// for bfs traversal
	public parent 	  : LGraphNode; // for bfs traversal
	public parentId   : number;
	public color 	  : number;

	public vContainer : PIXI.Container;
	public vGraphics  : PIXI.Graphics;
	public vText	  : PIXI.Text;

	public priority : number;
	public inHeap : boolean;
	public d : number;

	// private **********
	private m_data : number;
	private m_edges : Array<LGraphEdge>;

	constructor( data: number, id : number ) 
	{
		this.inHeap = false;
		this.priority = -1;
		this.d = 1000000;

		this.level = -1;
		this.parent = null;
		this.color = 0xffffff;

		this.m_data = data;
		this.m_edges = new Array<LGraphEdge>();
		this.id = id;

		this.vContainer = new PIXI.Container();
		this.vGraphics = new PIXI.Graphics();
		let _style : PIXI.TextStyle = new PIXI.TextStyle(
											{
												fontFamily : 'Arial',
												fontSize : 10
											} );
		this.vText = new PIXI.Text( '' + this.id, _style );
		this.vText.anchor.set( 0.5 );

		this.vGraphics.lineStyle( 1, 0x0000ff, 1 );
		this.vGraphics.beginFill( this.color, 0.75 );
		this.vGraphics.drawCircle( 0, 0, 10 );

		this.vContainer.addChild( this.vGraphics );
		this.vContainer.addChild( this.vText );
	}

	public free() : void
	{
		for ( var q = 0; q < this.m_edges.length; q++ )
		{
			this.m_edges[q].free();
			this.m_edges[q] = null;
		}

		this.vContainer.removeChild( this.vGraphics );
		this.vContainer.removeChild( this.vText );

		this.vGraphics = null;
		this.vText = null;
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

	public graphics_redraw() : void
	{

		this.vGraphics.clear();
		this.vGraphics.lineStyle( 1, 0x0000ff, 1 );
		this.vGraphics.beginFill( this.color, 1 );
		this.vGraphics.drawCircle( 0, 0, 10 );

	}
}


class LGraph
{
	public vContainer : PIXI.Container;
	public vBackContainer : PIXI.Container;
	public vFrontContainer : PIXI.Container;
	
	// for "pretty" animations of traversals
	public visitBuff_dfs : Array<LGraphNode>;
	public visitBuff_bfs : Array<LGraphNode>;
	public visitCounter : number;
	public visitTimeout : any;

	public bfs_st_nodes : Array<LGraphNode>;
	public bfs_st_edges : Array<LGraphEdge>;
	public dfs_st_nodes : Array<LGraphNode>;
	public dfs_st_edges : Array<LGraphEdge>;

	// to choose the type of graph
	public isWeighted : boolean;
	public isDirected : boolean;

	private m_nNodes : number;
	private m_nodes : Array<LGraphNode>;

	constructor() 
	{
		this.m_nodes = new Array<LGraphNode>();
		this.vContainer = new PIXI.Container();
		this.vBackContainer = new PIXI.Container();
		this.vFrontContainer = new PIXI.Container();
		this.vContainer.addChild( this.vBackContainer );
		this.vContainer.addChild( this.vFrontContainer );
		this.visitCounter = 0;
		this.visitBuff_dfs = new Array<LGraphNode>();
		this.visitBuff_bfs = new Array<LGraphNode>();
		this.m_nNodes = 0;

		this.bfs_st_nodes = new Array<LGraphNode>();
		this.bfs_st_edges = new Array<LGraphEdge>();
		this.dfs_st_nodes = new Array<LGraphNode>();
		this.dfs_st_edges = new Array<LGraphEdge>();

		this.isWeighted = false;
		this.isDirected = false;
	}

	public free() : void
	{
		for ( var q = 0; q < this.m_nNodes; q++ )
		{
			this.m_nodes[q].free();
			this.m_nodes[q] = null;
		}

		this.visitBuff_bfs = null;
		this.visitBuff_dfs = null;

		this.vContainer.removeChild( this.vBackContainer );
		this.vContainer.removeChild( this.vFrontContainer );
		this.vBackContainer = null;
		this.vFrontContainer = null;
	}

	public insertNode( nData : number, nId : number ) : LGraphNode
	{
		let _node : LGraphNode = new LGraphNode( nData, nId );

		this.m_nodes.push( _node );
		this.m_nNodes++;

		// Graphics part
		this.vFrontContainer.addChild( _node.vContainer );

		return _node;
	}

	public numNodes() : number
	{
		return this.m_nNodes;
	}

	public nodes() : Array<LGraphNode>
	{
		return this.m_nodes;
	}

	public insertEdge( nFrom: LGraphNode, nTo: LGraphNode, eData: number ) : Array<LGraphEdge>
	{
		let _edges_from : Array<LGraphEdge> = nFrom.getEdges();

		for ( var q = 0; q < _edges_from.length; q++ )
		{
			if ( ( _edges_from[q].to() ).id == nTo.id )
			{
				return;
			}
		}

		let _edge_direct : LGraphEdge = new LGraphEdge( eData, nFrom, nTo );
		nFrom.addEdge( _edge_direct );
		_edge_direct.isWeighted = this.isWeighted;

		let _edge_reverse : LGraphEdge = new LGraphEdge( eData, nTo, nFrom );
		nTo.addEdge( _edge_reverse );
		_edge_reverse.isWeighted = this.isWeighted;

		// Graphics part
		this.vBackContainer.addChild( _edge_direct.vContainer );
		//this.vBackContainer.addChild( _edge_reverse.vContainer ); // the reverse is just kept for safe delete

		return [_edge_direct, _edge_reverse];
	}

	public resetGraphics() : void
	{
		for ( var q = 0; q < this.m_nodes.length; q++ )
		{
			this.m_nodes[q].color = 0xffffff;
			this.m_nodes[q].graphics_redraw();
			let _edges : Array<LGraphEdge> = this.m_nodes[q].getEdges();
			for ( var p = 0; p < _edges.length; p++ )
			{
				_edges[p].color = 0x0000ff;
				_edges[p].graphics_redraw();
			}
		}

	}

	public test_priorityQueue() : void
	{
		for ( var q = 0; q < this.m_nodes.length; q++ )
		{
			this.m_nodes[q].priority = ( this.m_nodes[q].id + 1 ) * 10;
		}

		let _pq : LPriorityQueue = new LPriorityQueue( this.m_nodes );

		console.log( 'test pq' );
		while ( _pq.isEmpty() == false )
		{
			let _minNode : LGraphNode = _pq.extractMin();
			console.log( 'p: ' + _minNode.priority );
		}
	}

	public mst_prim() : LGraph
	{
		let _mst : LGraph = new LGraph();
		_mst.isWeighted = true;

		for ( var q = 0; q < this.m_nodes.length; q++ )
		{
			this.m_nodes[q].parent = null;
			this.m_nodes[q].parentId = -1;
			this.m_nodes[q].priority = 1000000;
		}

		// pick the start vertex as the first vertex
		this.m_nodes[0].priority = 0;

		let _pq : LPriorityQueue = new LPriorityQueue( this.m_nodes );

		while ( _pq.isEmpty() == false )
		{
			let _lNext : LGraphNode = _pq.getMin();
			// Add this to the tree
			let _lTreeNode : LGraphNode = _mst.insertNode( _lNext.getData(), _lNext.id );
			if ( _lNext.parentId != -1 )
			{
				let _nodes : Array<LGraphNode> = _mst.nodes();
				_mst.insertEdge( _nodes[_lTreeNode.id], _nodes[_lNext.parentId], _lNext.priority );
			}

			let _edges : Array<LGraphEdge> = _lNext.getEdges();
			for ( var q = 0; q < _edges.length; q++ )
			{
				let _v : LGraphNode = _edges[q].to();
				if ( _v.inHeap )
				{
					// relaxation

					if ( _edges[q].data() < _v.priority )
					{
						_v.priority = _edges[q].data();
						_v.parent = _lNext;
						_v.parentId = _lNext.id;
					}
				}
			}

			_pq.extractMin();
		}


		return _mst;
	}

	public spanningTree_bfs() : void
	{
		this.resetGraphics();
		// just pick the first node as start
		this.bfs( this.m_nodes[0] );
		// show bfs-spanning tree in the graph
		for ( var q = 0; q < this.bfs_st_nodes.length; q++)
		{
			this.bfs_st_nodes[q].color = 0x00ff00;
			this.bfs_st_nodes[q].graphics_redraw();
		}

		for ( var q = 0; q < this.bfs_st_edges.length; q++ )
		{
			this.bfs_st_edges[q].color = 0x00ff00;
			this.bfs_st_edges[q].graphics_redraw();
		}
	}

	public spanningTree_dfs() : void
	{
		this.resetGraphics();
		// just pick the first node as start
		this.dfs();
		// show bfs-spanning tree in the graph
		for ( var q = 0; q < this.dfs_st_nodes.length; q++)
		{
			this.dfs_st_nodes[q].color = 0x00ff00;
			this.dfs_st_nodes[q].graphics_redraw();
		}

		for ( var q = 0; q < this.dfs_st_edges.length; q++ )
		{
			this.dfs_st_edges[q].color = 0x00ff00;
			this.dfs_st_edges[q].graphics_redraw();
		}
	}

	public bfs( startNode : LGraphNode ) : void
	{
		for ( var q = 0; q < this.m_nNodes; q++ )
		{
			this.m_nodes[q].level = -1;
			this.m_nodes[q].parent = null;
		}

		// initialize animation stuff
		this.visitBuff_bfs = new Array<LGraphNode>();
		this.bfs_st_nodes = new Array<LGraphNode>();
		this.bfs_st_edges = new Array<LGraphEdge>();

		let _frontier : Array<LGraphNode> = new Array<LGraphNode>();
		let _iLevel : number = 1;
		_frontier.push( startNode );
		startNode.level = 0;
		this.visitBuff_bfs.push( startNode );
		this.bfs_st_nodes.push( startNode );

		while ( _frontier.length != 0 )
		{

			let _next : Array<LGraphNode> = new Array<LGraphNode>();

			let _color : number = Math.floor( 0xffffff * Math.random() );

			for ( var q = 0; q < _frontier.length; q++ )
			{
				let _u : LGraphNode = _frontier[q];
				let _edges : Array<LGraphEdge> = _u.getEdges();
				for ( var e = 0; e < _edges.length; e++ )
				{
					let _v : LGraphNode = _edges[e].to();
					if ( _v.level == -1 )
					{
						_v.level = _iLevel;
						_v.parent = _u;
						_next.push( _v );

						// Graphics stuff
						_v.color = _color;
						_v.graphics_redraw();
						// animation stuff
						this.visitBuff_bfs.push( _v );

						// spanning tree stuff
						this.bfs_st_nodes.push( _v );
						this.bfs_st_edges.push( _edges[e] );
					}
				}
			}

			_frontier = _next;
			_iLevel++;
		}
	}

	public animate_bfs_start() : void
	{
		for ( var q = 0; q < this.visitBuff_bfs.length; q++ )
		{
			this.visitBuff_bfs[q].vContainer.visible = false;
		}

		this.visitCounter = 0;

		this.visitTimeout = setInterval( this.animate_bfs_step, 1000, this );
	}

	public animate_bfs_step( self : LGraph ) : void
	{
		self.visitBuff_bfs[self.visitCounter].vContainer.visible = true;
		self.visitCounter++;
		if ( self.visitCounter == self.visitBuff_bfs.length )
		{
			self.animate_bfs_stop();
		}
	}

	public animate_bfs_stop() : void
	{
		this.visitCounter = 0;
		clearInterval( this.visitTimeout );
	}

	public dfs() : void
	{
		// clean animation stuff
		this.visitBuff_dfs = new Array<LGraphNode>();
		this.dfs_st_nodes = new Array<LGraphNode>();
		this.dfs_st_edges = new Array<LGraphEdge>();

		// Initialize dfs stuff for all nodes
		for ( var q = 0; q < this.m_nodes.length; q++ )
		{
			this.m_nodes[q].parent = null;
		}

		for ( var q = 0; q < this.m_nodes.length; q++ )
		{
			if ( this.m_nodes[q].parent == null )
			{
				this.m_nodes[q].parent = this.m_nodes[q];
				this.visitBuff_dfs.push( this.m_nodes[q] );
				this.dfs_st_nodes.push( this.m_nodes[q] );
				this.dfs_visit( this.m_nodes[q] );
			}
		}
				
	}

	public dfs_visit( u : LGraphNode ) : void
	{
		let _edges : Array<LGraphEdge> = u.getEdges();
		for ( var q = 0; q < _edges.length; q++ )
		{
			let _v : LGraphNode = _edges[q].to();
			if ( _v.parent == null )
			{
				_v.parent = u;
				// animation stuff
				this.visitBuff_dfs.push( _v );
				this.dfs_st_nodes.push( _v );
				this.dfs_st_edges.push( _edges[q] );
				this.dfs_visit( _v );
			}
		}
	}

	public animate_dfs_start() : void
	{
		for ( var q = 0; q < this.visitBuff_dfs.length; q++ )
		{
			this.visitBuff_dfs[q].vContainer.visible = false;
		}

		this.visitCounter = 0;

		this.visitTimeout = setInterval( this.animate_dfs_step, 1000, this );
	}

	public animate_dfs_step( self : LGraph ) : void
	{
		self.visitBuff_dfs[self.visitCounter].vContainer.visible = true;
		self.visitCounter++;
		if ( self.visitCounter == self.visitBuff_dfs.length )
		{
			self.animate_dfs_stop();
		}
	}

	public animate_dfs_stop() : void
	{
		this.visitCounter = 0;
		clearInterval( this.visitTimeout );
	}

	// Minimum spanning trees code

	public minimumSpanningTree_prim() : LGraph
	{
		let _mst : LGraph = new LGraph();

		



		return _mst;
	}

	// single source shortest paths

	public dijkstra
}


class LGraphMat
{
	public vContainer : PIXI.Container;
	public vGraphics : PIXI.Graphics;
	public vNodes : Array<PIXI.Point>;

	private m_mat 		: Array<Array<number>>;
	private m_nNodes	: number;

	constructor( nNodes : number )
	{
		this.m_nNodes = nNodes;
		this.m_mat = new Array<Array<number>>();
		for ( var q = 0; q < this.m_nNodes; q++ )
		{
			let _row : Array<number> = new Array<number>();
			for ( var p = 0; p < this.m_nNodes; p++ )
			{
				_row.push( 0 );
			}
			this.m_mat.push( _row );
		}

		this.vNodes = new Array<PIXI.Point>();
		for ( var q = 0; q < this.m_nNodes; q++ ) 
		{
			this.vNodes.push( new PIXI.Point() );
		}

		this.vContainer = new PIXI.Container();
		this.vGraphics = new PIXI.Graphics();
		this.vContainer.addChild( this.vGraphics );
	}

	public free() : void
	{
		this.vContainer.removeChild( this.vGraphics );
		this.vGraphics.clear();
		this.vGraphics = null;

		this.vNodes = null;
		this.m_mat = null;
	}

	public insertEdge( id1 : number, id2 : number, w : number ) : void
	{
		if ( id1 < 0 || id1 > this.m_nNodes - 1 )
		{
			return;
		}
		if ( id2 < 0 || id2 > this.m_nNodes - 1 )
		{
			return;
		}

		this.m_mat[id1][id2] = w;
		this.m_mat[id2][id1] = w;

	}		

	public graphics_redrawGraph() : void
	{
		this.vGraphics.clear();

		for ( var q = 0; q < this.m_nNodes; q++ )
		{
			for ( var p = q + 1; p < this.m_nNodes; p++ )
			{
				if ( this.m_mat[q][p] != 0 )
				{
					this.vGraphics.lineStyle( 1, 0x0000ff, 1 );
					this.vGraphics.beginFill( 0xffffff, 0.75 );
					this.vGraphics.drawCircle( this.vNodes[q].x, this.vNodes[q].y, 10 );
					this.vGraphics.drawCircle( this.vNodes[p].x, this.vNodes[p].y, 10 );
					this.vGraphics.beginFill( 0x0000ff );
					this.vGraphics.lineStyle( 1, 0x0000ff, 1 );
					this.vGraphics.moveTo( this.vNodes[q].x, this.vNodes[q].y );
					this.vGraphics.lineTo( this.vNodes[p].x, this.vNodes[p].y );
				}
			}
		}
	}
}