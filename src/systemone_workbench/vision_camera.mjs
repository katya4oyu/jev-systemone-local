// One camera frame per completed inference; stopping invalidates late work.
export function captureFrame(video) {
  if (!video.videoWidth || !video.videoHeight) {
    throw new Error('カメラ映像を読み取れません。再度開始してください。');
  }
  const scale = Math.min(1, 1024 / Math.max(video.videoWidth, video.videoHeight));
  const canvas = document.createElement('canvas');
  canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
  canvas.height = Math.max(1, Math.round(video.videoHeight * scale));
  canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
  return new Promise((resolve, reject) => {
    canvas.toBlob(blob => blob ? resolve(blob) : reject(new Error('カメラ画像を作成できません。')), 'image/jpeg', .85);
  });
}

export function createCamera({
  video, onFrame, onState, onError,
  getMedia = constraints => navigator.mediaDevices.getUserMedia(constraints),
  capture = captureFrame,
}) {
  let generation = 0;
  let active = false;
  let stream = null;

  const stop = () => {
    if (!active) return;
    active = false;
    generation += 1;
    if (stream) stream.getTracks().forEach(track => track.stop());
    stream = null;
    video.pause();
    video.srcObject = null;
    onState('idle');
  };

  const start = async () => {
    if (active) return;
    active = true;
    const currentGeneration = ++generation;
    const isCurrent = () => active && generation === currentGeneration;
    onState('starting');
    try {
      const acquired = await getMedia({
        audio: false,
        video: { facingMode: { ideal: 'user' }, width: { ideal: 640 }, height: { ideal: 480 } },
      });
      if (!isCurrent()) {
        acquired.getTracks().forEach(track => track.stop());
        return;
      }
      stream = acquired;
      for (const track of stream.getTracks()) {
        track.addEventListener('ended', () => {
          if (!isCurrent()) return;
          stop();
          onError(new Error('カメラの接続が切れました。再度開始してください。'));
        }, { once: true });
      }
      video.srcObject = stream;
      await video.play();
      if (!isCurrent()) return;
      onState('running');
      while (isCurrent()) {
        const frame = await capture(video);
        if (!isCurrent()) break;
        await onFrame(frame, isCurrent);
      }
    } catch (error) {
      if (!isCurrent()) return;
      stop();
      onError(error);
    }
  };

  return { start, stop, get active() { return active; } };
}
