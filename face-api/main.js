document.addEventListener('DOMContentLoaded', async () => {
  const video = document.getElementById('video');
  const canvas = document.getElementById('overlay');

  if (typeof faceapi === 'undefined') {
    console.error('❌ faceapi no está definido');
    return;
  }

  // ✅ Cargar todos los modelos necesarios
  await Promise.all([
    faceapi.nets.tinyFaceDetector.loadFromUri('face-api/models'),
    faceapi.nets.faceLandmark68Net.loadFromUri('face-api/models'),
    faceapi.nets.faceRecognitionNet.loadFromUri('face-api/models'),
    faceapi.nets.ssdMobilenetv1.loadFromUri('face-api/models') // NECESARIO para descriptores
  ]);
  console.log('✅ Todos los modelos cargados');

  // 📷 Iniciar cámara
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: {} });
    video.srcObject = stream;
  } catch (err) {
    console.error('❌ Error al acceder a la cámara', err);
    return;
  }

  // 🧠 Cargar rostros etiquetados
  const labeledFaceDescriptors = await loadLabeledImages();
  const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6);

  // 🟦 Cuando el video comienza o está listo
  function startRecognition() {
    const displaySize = { width: video.width, height: video.height };
    faceapi.matchDimensions(canvas, displaySize);
    setInterval(async () => {
      const detections = await faceapi
        .detectAllFaces(video, new faceapi.SsdMobilenetv1Options())
        .withFaceLandmarks()
        .withFaceDescriptors();
      const resizedDetections = faceapi.resizeResults(detections, displaySize);
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      resizedDetections.forEach(detection => {
        const bestMatch = faceMatcher.findBestMatch(detection.descriptor);
        const box = detection.detection.box;
        const drawBox = new faceapi.draw.DrawBox(box, { label: bestMatch.toString() });
        drawBox.draw(canvas);
      });
    }, 100);
  }

  // Iniciar reconocimiento cuando el video esté listo
  if (video.readyState >= 2) {
    startRecognition();
  } else {
    video.addEventListener('play', startRecognition);
    video.addEventListener('loadeddata', startRecognition);
  }
});

// 📂 Cargar imágenes etiquetadas
async function loadLabeledImages() {
  const labels = ['Luke', 'Nico']; // carpetas con imágenes
  // Ajusta la cantidad de imágenes por persona
  const imagesPerLabel = { Luke: 8, Nico: 7 };
  return Promise.all(
    labels.map(async label => {
      const descriptions = [];
      const numImages = imagesPerLabel[label] || 2;
      for (let i = 1; i <= numImages; i++) {
        try {
          const img = await faceapi.fetchImage(`face-api/labeled_images/${label}/${i}.jpg`);
          const detection = await faceapi
            .detectSingleFace(img)
            .withFaceLandmarks()
            .withFaceDescriptor();
          if (detection) {
            descriptions.push(detection.descriptor);
          } else {
            console.warn(`No se detectó rostro en ${label}/${i}.jpg`);
          }
        } catch (e) {
          console.error(`Error cargando imagen ${label}/${i}.jpg`, e);
        }
      }
      if (descriptions.length === 0) {
        console.warn(`No se encontraron descriptores para ${label}`);
      }
      return new faceapi.LabeledFaceDescriptors(label, descriptions);
    })
  );
}
