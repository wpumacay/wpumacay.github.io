
/// <reference path="LGraphs.ts"/>


class LHeap
{

	public m_arr : Array<LGraphNode>;


	constructor( arr : Array<LGraphNode> )
	{
		this.m_arr = new Array<LGraphNode>();
		for ( var q = 0; q < arr.length; q++ )
		{
			arr[q].inHeap = true;
			this.m_arr.push( arr[q] );
		}
	}

	public extractMinimum() : LGraphNode
	{
		if ( this.m_arr.length < 1 )
		{
			return null;
		}

		if ( this.m_arr.length == 1 )
		{
			return this.m_arr.shift();
		}

		let _min : LGraphNode = this.m_arr[0];
		this.m_arr[0] = this.m_arr.pop();
		this.minHeapify( 0 );

		_min.inHeap = false;
		return _min;
	}

	public getMin() : LGraphNode
	{
		return this.m_arr[0];
	}

	public isEmpty() : boolean
	{
		return this.m_arr.length < 1;
	}

	public static build_maxHeap( arr : Array<LGraphNode> ) : LHeap
	{
		let _res : LHeap = new LHeap( arr );

		for ( var q = Math.ceil( arr.length / 2 ); q >= 1; q-- )
		{
			_res.maxHeapify( q );
		}

		return _res;
	}

	public static build_minHeap( arr : Array<LGraphNode> ) : LHeap
	{
		let _res : LHeap = new LHeap( arr );

		for ( var q = Math.ceil( arr.length / 2 ); q >= 1; q-- )
		{
			_res.minHeapify( q );
		}

		return _res;
	}

	public maxHeapify( i : number ) : void
	{
		let _l : number = this.left( i );
		let _r : number = this.right( i );
		let _largest : number = i;

		if ( _l < this.m_arr.length &&
			 this.m_arr[_l].priority < this.m_arr[i].priority )
		{
			_largest = _l;
		}
		else
		{
			_largest = i;
		}

		if ( _r < this.m_arr.length &&
			 this.m_arr[_r].priority < this.m_arr[_largest].priority )
		{
			_largest = _r;
		}

		if ( _largest != i )
		{
			this.swap( i, _largest );
			this.maxHeapify( _largest );
		}
	}

	public minHeapify( i : number ) : void
	{
		let _l : number = this.left( i );
		let _r : number = this.right( i );
		let _smallest : number = i;

		if ( _l < this.m_arr.length &&
			 this.m_arr[_l].priority < this.m_arr[_smallest].priority )
		{
			_smallest = _l;
		}

		if ( _r < this.m_arr.length &&
			 this.m_arr[_r].priority < this.m_arr[_smallest].priority )
		{
			_smallest = _r;
		}

		if ( _smallest != i )
		{
			this.swap( i, _smallest );
			this.maxHeapify( _smallest );
		}
	}

	private left( i : number ) : number
	{
		return 2 * i + 1;
	}

	private right( i : number ) : number
	{
		return 2 * i + 2;
	}

	private parent( i : number ) : number
	{
		return Math.floor( ( i - 1 ) / 2 );
	}

	private swap( i : number, j : number )
	{
		let _tmp : LGraphNode = this.m_arr[i];
		this.m_arr[i] = this.m_arr[j];
		this.m_arr[j] = _tmp;
	}

}



class LPriorityQueue
{


	public m_heap : LHeap;

	constructor( nodes : Array<LGraphNode> )
	{
		this.m_heap = LHeap.build_minHeap( nodes );
	}

	public extractMin() : LGraphNode
	{
		let _res : LGraphNode = this.m_heap.extractMinimum();

		return _res;
	}

	public getMin() : LGraphNode
	{
		return this.m_heap.getMin();
	}

	public isEmpty() : boolean
	{
		return this.m_heap.isEmpty();
	}

}