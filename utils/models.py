import tensorflow as tf
import torch
import numpy as np
from tqdm import tqdm
import mxnet as mx
from mxnet import ndarray as nd


def batch_wise_inference(imgs: np.ndarray, inference_fn: callable, batch_size: int) -> np.ndarray:
    """Function to perform inference in batch-wise

    :param imgs: List of images
    :param inference_fn: The inference function for one single image
    :param batch_size: size of the batches
    :return: List of embeddings
    """

    embs = []
    for i in tqdm(range(0, imgs.shape[0], batch_size), desc="Inference:"):
        embs.append(inference_fn(imgs[i : i + batch_size]))
    return np.concatenate(embs)


class ArcFaceOctupletLoss:
    """ArcFaceOctupletLoss Model
    https://github.com/martlgap/octuplet-loss
    """

    def __init__(self, batch_size: int = 64) -> None:
        self.model = tf.keras.models.load_model("../models/ArcFaceOctupletLoss.tf")
        self.batch_size = batch_size

    @staticmethod
    def preprocess(img) -> np.ndarray:
        if img.ndim != 4:
            img = np.expand_dims(img, axis=0)
        return img

    def single_inference(self, img) -> np.ndarray:
        return self.model(self.preprocess(img))

    def __call__(self, imgs) -> np.ndarray:
        return batch_wise_inference(imgs, self.single_inference, self.batch_size)


class FaceTransformerOctupletLoss:
    """FaceTransformerOctupletLoss Model
    https://github.com/martlgap/octuplet-loss
    """

    def __init__(self, batch_size: int = 32) -> None:
        self.device = torch.device("cuda")  # or cuda
        self.model = torch.load("../models/FaceTransformerOctupletLoss.pt", map_location=self.device)
        self.model.eval()
        self.batch_size = batch_size

    def preprocess(self, img) -> np.ndarray:
        if img.ndim != 4:
            img = np.expand_dims(img, axis=0)
        img = torch.from_numpy(np.transpose(img, [0, 3, 1, 2]).astype("float32") * 255).clamp(0.0, 255.0).to(self.device)
        return img

    def single_inference(self, img) -> np.ndarray:
        return self.model(self.preprocess(img)).cpu().detach().numpy()

    def __call__(self, imgs) -> np.ndarray:
        return batch_wise_inference(imgs, self.single_inference, self.batch_size)


class ProdPoly:
    """ProdPoly Model
    https://github.com/grigorisg9gr/polynomial_nets
    """

    def __init__(self, batch_size: int = 32) -> None:
        sym, arg_params, aux_params = mx.model.load_checkpoint("../models/TPAMI2020-PiNet/polynet50/model", 1)
        all_layers = sym.get_internals()
        sym = all_layers["fc1_output"]
        self.model = mx.mod.Module(symbol=sym, context=mx.gpu(0), label_names=None)
        self.model.bind(data_shapes=[("data", (batch_size, 3, 112, 112))])
        self.model.set_params(arg_params, aux_params)
        self.batch_size = batch_size

    def preprocess(self, img) -> np.ndarray:
        if img.ndim != 4:
            img = np.expand_dims(img, axis=0)
        img = (np.transpose(img, (0, 3, 1, 2)) * 255).astype(np.uint8).astype(np.float32)
        return img

    def single_inference(self, img) -> np.ndarray:
        db = mx.io.DataBatch(
            data=(nd.from_numpy(np.ascontiguousarray(self.preprocess(img))),),
            label=(nd.ones((self.batch_size,)),),
        )
        self.model.forward(db, is_train=False)
        net_out = self.model.get_outputs()
        return net_out[0].asnumpy()

    def __call__(self, imgs) -> np.ndarray:
        return batch_wise_inference(imgs, self.single_inference, self.batch_size)
