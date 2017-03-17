var {NativeModules} = require('react-native');
var RNCaffe  = NativeModules.RNCaffe;

module.exports 	= {
  faceDetect(path) {
    return RNCaffe.faceDetect(path);
  },
  setup4(model,weight,mean,label){
    return RNCaffe.setup4(model,weight,mean,label);
  },
  setup3(model,weight,mean){
    return RNCaffe.setup3(model,weight,mean);
  },
  setup2(model,weight){
    return RNCaffe.setup2(model,weight);
  },
  seeImage(image){
    return RNCaffe.seeImage(image);
  },
  resizeImage(pathIn,pathOut){
    return RNCaffe.resizeImage(pathIn,pathOut);
  },
  run_model(_model,_weight,_image){
    return RNCaffe.runModel(_model,_weight,_image);
  },
}
