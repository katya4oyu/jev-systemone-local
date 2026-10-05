import test from 'node:test';
import assert from 'node:assert/strict';
import { createCamera } from '../src/systemone_workbench/vision_camera.mjs';

const deferred = () => {
  let resolve;
  const promise = new Promise(done => { resolve = done; });
  return { promise, resolve };
};
const settle = () => new Promise(resolve => setImmediate(resolve));
const fixture = (options = {}) => {
  const states = [], errors = [], constraints = [];
  const track = { stopped: false, stop() { this.stopped = true; }, addEventListener() {} };
  const stream = { getTracks: () => [track] };
  const video = { srcObject: null, play: async () => {}, pause() {} };
  const camera = createCamera({
    video,
    getMedia: async value => { constraints.push(value); return stream; },
    capture: async () => new Blob(['frame'], { type: 'image/jpeg' }),
    onState: state => states.push(state), onError: error => errors.push(error),
    ...options,
  });
  return { camera, video, track, states, errors, constraints, stream };
};

test('frames are serialized; stop releases the track and invalidates an in-flight result', async () => {
  const first = deferred(), second = deferred();
  const current = [];
  const f = fixture({ onFrame: async (_blob, isCurrent) => {
    current.push(isCurrent);
    await (current.length === 1 ? first.promise : second.promise);
  } });
  const running = f.camera.start();
  await settle();
  assert.equal(current.length, 1);
  await f.camera.start();
  assert.equal(f.constraints.length, 1);
  assert.equal(f.constraints[0].audio, false);
  first.resolve();
  await settle();
  assert.equal(current.length, 2);
  f.camera.stop();
  assert.equal(f.track.stopped, true);
  assert.equal(f.video.srcObject, null);
  assert.equal(current[1](), false);
  second.resolve();
  await running;
  assert.equal(current.length, 2);
  assert.deepEqual(f.states, ['starting', 'running', 'idle']);
  assert.deepEqual(f.errors, []);
});

test('stopping during permission request releases a late stream without attaching it', async () => {
  const permission = deferred();
  let frameCount = 0;
  const f = fixture({ getMedia: () => permission.promise, onFrame: async () => { frameCount++; } });
  const running = f.camera.start();
  f.camera.stop();
  permission.resolve(f.stream);
  await running;
  assert.equal(f.track.stopped, true);
  assert.equal(f.video.srcObject, null);
  assert.equal(frameCount, 0);
  assert.deepEqual(f.states, ['starting', 'idle']);
});

test('denied permissions return to idle and can be retried', async () => {
  let attempt = 0;
  let f;
  f = fixture({
    getMedia: async () => {
      if (attempt++ === 0) throw new DOMException('denied', 'NotAllowedError');
      return f.stream;
    },
    onFrame: async () => f.camera.stop(),
  });
  await f.camera.start();
  assert.equal(f.camera.active, false);
  assert.equal(f.errors[0].name, 'NotAllowedError');
  await f.camera.start();
  assert.equal(f.track.stopped, true);
  assert.deepEqual(f.states, ['starting', 'idle', 'starting', 'running', 'idle']);
});

test('inference failure stops the camera instead of retrying indefinitely', async () => {
  let frameCount = 0;
  const f = fixture({ onFrame: async () => { frameCount++; throw new Error('inference failed'); } });
  await f.camera.start();
  assert.equal(frameCount, 1);
  assert.equal(f.track.stopped, true);
  assert.equal(f.camera.active, false);
  assert.equal(f.errors[0].message, 'inference failed');
});
