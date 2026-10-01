"""Run with python -m unittest discover -s tests -p 'test_temporal18.py'.

All fixtures use /tmp. Real assets are exercised separately by bounded GPU smoke runs.
"""

import copy
import importlib.util
import json
import math
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from PIL import Image

from src.config import load_typed_root_config
from src.dataset import get_dataset
from src.dataset.dataset_temporal18 import CAMERAS
from src.dataset.shims.patch_shim import apply_patch_shim
from src.dataset.temporal18.shim import with_eval_mask_crop
from src.evaluation.temporal18_metrics import compute_image_metrics, compute_group_pcc, metric_records
from src.evaluation.temporal18_reporting import EvaluationWriter, summarize
from src.model.model_wrapper import ModelWrapper
from src.model.model_wrapper_temporal18 import ModelWrapperTemporal18
from src.global_cfg import set_cfg


ROOT = Path(__file__).resolve().parents[1]
BASE_OVERRIDES = ['model.encoder.num_scales=2', 'model.encoder.upsample_factor=2',
                  'model.encoder.lowest_feature_resolution=4', 'model.encoder.monodepth_vit_type=vitb']


def config(name):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / 'config')):
        return compose(config_name='main', overrides=[f'+experiment={name}'])


class ConfigTests(unittest.TestCase):
    def test_all_configs_match_omniscene_base(self):
        with initialize_config_dir(version_base=None, config_dir=str(ROOT / 'config')):
            for resolution in ('112x200', '224x400'):
                original = compose(config_name='main', overrides=[f'+experiment=omniscene_{resolution}', *BASE_OVERRIDES])
                for dataset in ('pandaset', 'ddad'):
                    for suffix in ('', '_zero_shot'):
                        current = compose(config_name='main', overrides=[f'+experiment={dataset}_{resolution}{suffix}'])
                        typed = load_typed_root_config(current)
                        self.assertEqual(typed.dataset.name, dataset)
                        for section in ('model', 'loss', 'optimizer', 'trainer', 'data_loader'):
                            self.assertEqual(OmegaConf.to_container(current[section]), OmegaConf.to_container(original[section]))
                        train = OmegaConf.to_container(original.train)
                        train['use_dynamic_mask'] = False
                        self.assertEqual(train, OmegaConf.to_container(current.train))
                        self.assertEqual(current.dataset.image_shape, original.dataset.image_shape)
                        self.assertEqual(current.mode, 'test' if suffix else 'train')

    def test_training_methods_inherited(self):
        for method in ('training_step', 'configure_optimizers', 'on_validation_epoch_end'):
            self.assertIs(getattr(ModelWrapperTemporal18, method), getattr(ModelWrapper, method))

    def test_output_interpolation_and_checkpoint_parameter_coverage(self):
        with tempfile.TemporaryDirectory(dir='/tmp', prefix='volsplat-wrapper-') as tmp:
            cfg = config('ddad_112x200')
            cfg.output_dir = tmp
            typed = load_typed_root_config(cfg)
            set_cfg(cfg)
            encoder = torch.nn.Linear(2, 2)
            encoder.cfg = SimpleNamespace(shim_patch_size=4, downscale_factor=4)
            model = ModelWrapperTemporal18(typed.optimizer, typed.test, typed.train, encoder, None,
                                           torch.nn.Identity(), [], None, dataset_cfg=typed.dataset)
            self.assertEqual(model.test_cfg.output_path, Path(tmp) / 'metrics')
            self.assertEqual(set(model.state_dict()), {'encoder.weight', 'encoder.bias'})
            with self.assertRaisesRegex(RuntimeError, 'checkpoint/model mismatch'):
                model.load_state_dict({}, strict=False)
            model.load_state_dict(model.state_dict(), strict=False)
            self.assertTrue(model._weights_loaded)
            dataset = SimpleNamespace(resolution=(112, 200), evaluation_metadata=lambda: {'expected_bins': 1})
            metadata = model._metadata(dataset)
            self.assertEqual(metadata['evaluation_resolution'], [112, 192])
            self.assertIsNone(metadata['source_checkpoint'])


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='volsplat-temporal18-fixture-', dir='/tmp')
        self.root = Path(self.temporary.name)
        (self.root / 'bin_infos').mkdir()
        self.tokens = [f'bin{i:02}' for i in range(13)]
        (self.root / 'bins_train.json').write_text(json.dumps({'bins': self.tokens[:2]}))
        (self.root / 'bins_test.json').write_text(json.dumps({'bins': self.tokens}))
        sensors = {}
        for camera_index, camera in enumerate(CAMERAS):
            sensors[camera] = []
            for moment in range(3):
                index = camera_index * 3 + moment
                Image.fromarray(np.full((112, 200, 3), index * 10, np.uint8)).save(self.root / f'{index}.png')
                (self.root / f'{index}.json').write_text(json.dumps({'camera_intrinsic': [[100,0,100],[0,80,56],[0,0,1]]}))
                np.save(self.root / f'{index}.npy', np.full((112,200), index + 1, np.float32))
                transform = np.eye(4, dtype=np.float32)
                transform[0, 3] = index
                sensors[camera].append({'data_path': f'{index}.png', 'intrinsic_path': f'{index}.json',
                                       'depth_path': f'{index}.npy', 'sensor2lidar_transform': transform, 'camera': camera})
        for token in self.tokens:
            with (self.root / 'bin_infos' / f'{token}.pkl').open('wb') as handle:
                pickle.dump({'scene_id': 'scene', 'sensor_info': sensors}, handle)
        mask_root = self.root / 'ego_masks' / 'vidar_v1'
        mask_root.mkdir(parents=True)
        mask = np.full((112,200),255,np.uint8)
        mask[:, :20] = 0
        Image.fromarray(mask).save(mask_root / 'mask.png')
        manifest = {'scene_camera_to_variant': {'scene': {c:'shared' for c in CAMERAS}},
                    'variants': {'shared': {'mask_id':'mask'}}, 'masks': {'mask': {'path':'mask.png'}}}
        (mask_root / 'manifest.json').write_text(json.dumps(manifest))

    def tearDown(self):
        self.temporary.cleanup()

    def dataset(self, name, stage):
        cfg = config(f'{name}_112x200')
        cfg.dataset.processed_root = str(self.root)
        return get_dataset(load_typed_root_config(cfg).dataset, stage, None)

    def test_splits_views_geometry_and_no_audit_manifests(self):
        for name in ('pandaset', 'ddad'):
            self.assertEqual(len(self.dataset(name,'train')),2)
            self.assertEqual(self.dataset(name,'val').bin_tokens,
                             [self.tokens[i] for i in np.linspace(0,12,10,dtype=int)])
            data = self.dataset(name,'test')
            self.assertEqual(len(data),13)
            sample = data[0]
            expected = [c*3+t for c in range(6) for t in (1,2)] + [c*3 for c in range(6)]
            forward_axis = 1 if name == 'ddad' else 0
            torch.testing.assert_close(sample['target']['extrinsics'][:,forward_axis,3], torch.tensor(expected,dtype=torch.float32))
            torch.testing.assert_close(sample['context']['extrinsics'][:,2,2],torch.ones(6))
            torch.testing.assert_close(sample['target']['rel_depth'][:,0,0],torch.tensor(expected,dtype=torch.float32)+1)
            torch.testing.assert_close(sample['context']['intrinsics'][0],torch.tensor([[.5,0,.5],[0,80/112,.5],[0,0,1]]))
            self.assertNotIn('rel_depth',sample['context'])
            self.assertNotIn('masks',sample['target'])
            train = self.dataset(name,'train')[0]
            self.assertNotIn('eval_mask',train['target'])
            self.assertNotIn('rel_depth',train['target'])

    def test_ddad_reference_frame_preserves_relative_geometry(self):
        from src.dataset.dataset_temporal18 import DatasetTemporal18

        # Camera-local right/down/forward expressed in DDAD's forward/left/up
        # world axes. Use nonzero translations so rotating only R cannot pass.
        center = np.array([[0, 0, 1, 2], [-1, 0, 0, -3],
                           [0, -1, 0, 1.5], [0, 0, 0, 1]], dtype=np.float32)
        path = self.root / 'bin_infos' / 'bin00.pkl'
        with path.open('rb') as handle:
            info = pickle.load(handle)
        for camera_index, camera in enumerate(CAMERAS):
            angle = camera_index * np.pi / 3
            yaw = np.array([[np.cos(angle), -np.sin(angle), 0, 0],
                            [np.sin(angle), np.cos(angle), 0, 0],
                            [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
            for moment, sensor in enumerate(info['sensor_info'][camera]):
                pose = yaw @ center
                pose[:3, 3] += np.array([moment * .7, camera_index * .1, -.2])
                sensor['sensor2lidar_transform'] = pose
        with path.open('wb') as handle:
            pickle.dump(info, handle)

        camera_point = torch.tensor([.25, -.5, 8., 1.])
        for stage in ('train', 'val', 'test'):
            dataset = self.dataset('ddad', stage)
            original = DatasetTemporal18(dataset.cfg, stage, None)[0]
            aligned = dataset[0]
            # Front looks along +Y and camera right along +X after conversion.
            torch.testing.assert_close(aligned['context']['extrinsics'][0,:3,2], torch.tensor([0.,1.,0.]))
            torch.testing.assert_close(aligned['context']['extrinsics'][0,:3,0], torch.tensor([1.,0.,0.]))
            # Every context-to-target transform and projected pixel stays equal.
            old_relative = original['target']['extrinsics'][:,None].inverse() @ original['context']['extrinsics'][None]
            new_relative = aligned['target']['extrinsics'][:,None].inverse() @ aligned['context']['extrinsics'][None]
            torch.testing.assert_close(new_relative, old_relative, atol=2e-6, rtol=1e-5)
            old_pixel = original['target']['intrinsics'][:,None] @ (old_relative @ camera_point)[...,:3,None]
            new_pixel = aligned['target']['intrinsics'][:,None] @ (new_relative @ camera_point)[...,:3,None]
            torch.testing.assert_close(new_pixel, old_pixel, atol=2e-6, rtol=1e-5)
            for side in ('context', 'target'):
                for key in original[side]:
                    if key != 'extrinsics':
                        torch.testing.assert_close(aligned[side][key], original[side][key], atol=0, rtol=0)
            torch.testing.assert_close(aligned['target']['extrinsics'][12:], aligned['context']['extrinsics'])
            self.assertEqual(dataset.evaluation_metadata()['camera_frame'], 'nuscenes_axes_x_right_y_forward_z_up')

    def test_ddad_crop_and_input_support(self):
        from torch.utils.data import default_collate
        batch = default_collate([self.dataset('ddad','test')[0]])
        original = copy.deepcopy(batch)
        shim = with_eval_mask_crop(lambda value: apply_patch_shim(value,16))
        cropped = shim(batch)
        self.assertEqual(cropped['target']['image'].shape,(1,18,3,112,192))
        torch.testing.assert_close(cropped['target']['eval_mask'],original['target']['eval_mask'][...,4:196])
        self.assertTrue(cropped['target']['eval_mask'][:,12:].all())
        self.assertFalse(cropped['target']['eval_mask'][:,:12,:,:16].any())
        torch.testing.assert_close(cropped['target']['rel_depth'],original['target']['rel_depth'][...,4:196])


class MetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        torch.manual_seed(42)
        cls.gt = torch.rand(18,3,32,48)
        cls.pred = (cls.gt + .1 * torch.randn_like(cls.gt)).clamp(0,1)
        cls.mask = torch.ones(18,32,48,dtype=torch.bool)
        cls.mask[:12,:,30:] = False

    def test_against_svfgs_reference_if_available(self):
        reference_path = Path('/home/dzp62442/Projects/SVF-GS/tools/metrics.py')
        if not reference_path.exists():
            self.skipTest('SVF-GS checkout not present')
        spec = importlib.util.spec_from_file_location('svfgs_metric_reference', reference_path)
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        from src.evaluation.temporal18_metrics import MASK_METRIC_CONFIG
        actual = compute_image_metrics(self.gt,self.pred,self.mask)
        expected = reference.compute_image_metrics(self.gt,self.pred,self.mask,MASK_METRIC_CONFIG)
        for name in actual:
            torch.testing.assert_close(actual[name],expected[name],rtol=1e-6,atol=1e-6)
        depth = torch.rand(18,32,48)
        predicted = depth * 2 + .1 * torch.randn_like(depth)
        for indices in (slice(None),slice(0,12),slice(12,18)):
            score,_ = compute_group_pcc(depth[indices],predicted[indices],self.mask[indices])
            self.assertAlmostEqual(score,reference.compute_eval_pcc(depth[indices],predicted[indices],self.mask[indices]).item(),places=6)

    def test_masks_preserve_input_and_do_not_mutate_predictions(self):
        original = self.pred.clone()
        masked = compute_image_metrics(self.gt,self.pred,self.mask)
        full = compute_image_metrics(self.gt,self.pred)
        for name in masked:
            torch.testing.assert_close(masked[name][12:],full[name][12:],rtol=0,atol=0)
        changed = self.pred.clone()
        changed[:12,:,:,30:] = 1 - changed[:12,:,:,30:]
        replaced = compute_image_metrics(self.gt,changed,self.mask)
        for name in masked:
            torch.testing.assert_close(masked[name],replaced[name],rtol=0,atol=0)
        torch.testing.assert_close(original,self.pred,rtol=0,atol=0)

    def test_undefined_metrics_do_not_raise(self):
        mask = torch.zeros_like(self.mask)
        values = compute_image_metrics(self.gt,self.pred,mask)
        self.assertTrue(all(torch.isnan(value).all() for value in values.values()))
        pcc,reason = compute_group_pcc(torch.ones(1,3,3),torch.ones(1,3,3))
        self.assertTrue(math.isnan(pcc))
        self.assertIn('constant',reason)

    def test_group_aggregation_and_reporting(self):
        batch = {'scene':['same_token'],'scene_id':['scene'],'sample_index':torch.tensor([0]),
                 'target':{'image':self.gt[None],'eval_mask':self.mask[None]}}
        rows = metric_records(batch,SimpleNamespace(color=self.pred[None],depth=None),'ddad_ego_novel12_v1')
        for name in ('psnr','ssim','lpips'):
            self.assertAlmostEqual(rows[0][name],(12*rows[1][name]+6*rows[2][name])/18,places=7)
        more = [{**row,'sample_index':1} for row in rows]
        with tempfile.TemporaryDirectory(dir='/tmp',prefix='volsplat-report-') as tmp:
            writer = EvaluationWriter(tmp,{'expected_bins':2})
            writer.append(rows+more+rows) # sampler padding, not duplicate token filtering
            summary = writer.finish()
            self.assertTrue(summary['complete'])
            self.assertEqual(summary['record_count'],6)
            self.assertEqual(summary['processed_bins'],2)
            self.assertIsNone(summary['groups']['all_18']['pcc'])
            self.assertEqual(len((Path(tmp)/'per_bin_metrics.csv').read_text().splitlines()),7)
            json.loads((Path(tmp)/'evaluation_summary.json').read_text())
        rows[0]['psnr'] = float('nan')
        summary = summarize(rows+more,{'expected_bins':2})
        self.assertTrue(summary['complete'])
        self.assertIsNone(summary['groups']['all_18']['psnr'])
        self.assertEqual(summary['groups']['all_18']['undefined_counts']['psnr'],1)


if __name__ == '__main__':
    unittest.main()
