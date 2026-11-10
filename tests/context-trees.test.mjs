import test from 'node:test';
import assert from 'node:assert/strict';
import {treePositions} from '../public/context-trees.js';
test('illustrative trees stay in fetched triangles and keep spacing and count bounds',()=>{
 const mesh={vertices:[[0,0,0],[40,0,0],[0,40,0]],faces:[[0,1,2]]};
 const points=treePositions([mesh]);assert.ok(points.length>0&&points.length<=8);
 for(const [x,y,z] of points){assert.ok(x>=0&&y>=0&&x+y<=40);assert.equal(z,0);}
 for(let i=0;i<points.length;i++)for(let j=0;j<i;j++)assert.ok(Math.hypot(points[i][0]-points[j][0],points[i][1]-points[j][1])>5);
 assert.deepEqual(treePositions([mesh]),points);
 assert.equal(treePositions([mesh],3).length,3);
});
test('missing green geometry creates no invented trees',()=>assert.deepEqual(treePositions([]),[]));
