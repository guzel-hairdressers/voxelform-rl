import test from 'node:test';
import assert from 'node:assert/strict';
import {sceneData,sceneOffset,voxelPosition} from '../public/viewer-data.js';
const mesh={vertices:[[11,21,0],[15,21,0],[15,24,6]],faces:[[0,1,2]],height:6};
const context={offset:[10,20,0],site_id:'test',city:'Test',tier:'S',siteArea:12,radius:150,lat:40,lon:2,perimeter:[[11,21],[15,21],[15,24]],site:mesh,buildings:[mesh],roads:[mesh],greenSpaces:[]};
test('the shared engine receives original metric coordinates for every scene layer',()=>{
 const state={shape:[4,8,10],context};const before=JSON.stringify(context);const scene=sceneData(state);
 for(const layer of [scene.site,...scene.buildings,...scene.roads])assert.deepEqual(layer.vertices,[[1,1,0],[5,1,0],[5,4,6]]);
 assert.deepEqual(scene.sitePerimeter,[[1,1],[5,1],[5,4]]);assert.equal(scene.radius,150);assert.equal(JSON.stringify(context),before);
});
test('voxel overlay coordinates align with the original engine X/up/negative-Y frame',()=>{
 const shape=[4,8,10],offset=sceneOffset({shape,context});
 assert.deepEqual(voxelPosition(0,shape,offset),[-9.5,1.5,19.5]);
 assert.deepEqual(voxelPosition(2*80+3*10+4,shape,offset),[-5.5,7.5,16.5]);
});
test('synthetic fixtures retain the allowed parcel rather than filling its bounding box',()=>{
 const state={shape:[1,2,2],context:null,allowed:[1,0,0,1],boxes:[]};const scene=sceneData(state);
 assert.equal(scene.site.faces.length,4);assert.equal(scene.siteArea,2);assert.equal(scene.buildings.length,0);
 assert.deepEqual(sceneOffset(state),[1,1,0]);
});
test('synthetic oriented boxes retain their heights and rotated dimensions',()=>{
 const state={shape:[1,2,2],context:null,allowed:[1,1,1,1],boxes:[[10,20,3,2,1,3,0,1]]};
 const building=sceneData(state).buildings[0];assert.equal(building.height,6);assert.equal(building.vertices.length,8);
 assert.deepEqual(building.vertices[0],[10,17,0]);assert.deepEqual(building.vertices[6],[8,21,6]);
});
test('rotated grid returns context and voxel centres to original world coordinates',()=>{
 const state={shape:[4,8,10],context:{...context,gridAngle:Math.PI/2}};
 const scene=sceneData(state),p=scene.buildings[0].vertices[0];
 assert.ok(Math.abs(p[0]+1)<1e-10);assert.ok(Math.abs(p[1]-1)<1e-10);
 const centre=voxelPosition(0,state.shape,context.offset,Math.PI/2);
 assert.ok(Math.abs(centre[0]-19.5)<1e-10);assert.equal(centre[1],1.5);assert.ok(Math.abs(centre[2]-9.5)<1e-10);
});
