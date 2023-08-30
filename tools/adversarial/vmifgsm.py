import torch
import torch.nn as nn

from .attack import Attack


class VMIFGSM(Attack):
    r"""
    VMI-FGSM in the paper 'Enhancing the Transferability of Adversarial Attacks through Variance Tuning
    [https://arxiv.org/abs/2103.15571], Published as a conference paper at CVPR 2021
    Modified from "https://github.com/JHL-HUST/VT"

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        decay (float): momentum factor. (Default: 1.0)
        steps (int): number of iterations. (Default: 5)
        N (int): the number of sampled examples in the neighborhood. (Default: 20)
        beta (float): the upper bound of neighborhood. (Default: 3/2)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.VMIFGSM(model, eps=8/255, steps=5, decay=1.0, N=20, beta=3/2)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, args):
        super().__init__("VMIFGSM", model)
        self.eps = args.eps
        self.steps = args.steps
        self.decay = args.decay
        self.alpha = args.alpha
        self.N = args.N
        self.beta = args.beta
        self._supported_mode = ['default', 'targeted']



    def forward(self, data):
        r"""
        Overridden.
        """
        images = data['img'][0].data[0].clone().detach().to(self.device)
        ub,lb = torch.max(images.view(3,-1),dim=1).values,torch.min(images.view(3,-1),dim=1).values
        eps = self.eps * torch.max(ub - lb )
        alpha = self.alpha * torch.max(ub - lb)
        momentum = torch.zeros_like(images).detach().to(self.device)

        v = torch.zeros_like(images).detach().to(self.device)
        adv_images = images.clone().detach()
        
        new_data = {}
        new_data['img_metas'] = data['img_metas'][0].data[0]
        for _ in range(self.steps):
            adv_images.requires_grad = True

            new_data['img'] = adv_images

            if 'gt_masks' in data.keys():
                losses = self.model(**new_data, return_loss=True,gt_bboxes=data['gt_bboxes'][0].data[0],
                                gt_labels=data['gt_labels'][0].data[0], gt_masks=  data['gt_masks'][0].data[0])
                loss_cls = sum(losses[_loss].mean() for _loss in losses.keys() if 'cls' in _loss and isinstance(losses[_loss],torch.Tensor))
            else:
                losses = self.model(**new_data, return_loss=True,gt_bboxes=data['gt_bboxes'][0].data[0],
                                gt_labels=data['gt_labels'][0].data[0])
                loss_cls = sum(_loss.mean() for _loss in losses['loss_cls'])

            self.model.zero_grad()
            loss_cls= loss_cls* (-1.0)
            loss_cls.backward()
            adv_grad = adv_images.grad.data

            grad = (adv_grad + v) / torch.mean(torch.abs(adv_grad + v), dim=(1,2,3), keepdim=True)
            grad = grad + momentum * self.decay
            momentum = grad

            # Calculate Gradient Variance
            GV_grad = torch.zeros_like(images).detach().to(self.device)
            nbi_data = {}
            nbi_data['img_metas'] = data['img_metas'][0].data[0]
            for _ in range(self.N):
                neighbor_images = adv_images.detach() + \
                                  torch.randn_like(images).uniform_(-eps*self.beta, eps*self.beta)
                neighbor_images.requires_grad = True
                nbi_data['img']=neighbor_images
                if 'gt_masks' in data.keys():
                    losses = self.model(**nbi_data, return_loss=True, gt_bboxes=data['gt_bboxes'][0].data[0],
                                        gt_labels=data['gt_labels'][0].data[0], gt_masks=data['gt_masks'][0].data[0])
                    loss_cls = sum(losses[_loss].mean() for _loss in losses.keys() if
                                   'cls' in _loss and isinstance(losses[_loss], torch.Tensor))
                else:
                    losses = self.model(**nbi_data, return_loss=True, gt_bboxes=data['gt_bboxes'][0].data[0],
                                        gt_labels=data['gt_labels'][0].data[0])
                    loss_cls = sum(_loss.mean() for _loss in losses['loss_cls'])

                self.model.zero_grad()
                loss_cls = loss_cls * (-1.0)
                loss_cls.backward()
                GV_grad = neighbor_images.grad.data
            # obtaining the gradient variance
            v = GV_grad / self.N - adv_grad

            adv_images = adv_images.detach() - alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-eps, max=eps)
            for chn in range(adv_images.shape[1]):
                adv_images[:,chn:chn+1,:,:] = torch.clamp(images[:,chn:chn+1,:,:] + delta[:,chn:chn+1,:,:], min=lb[chn], max=ub[chn]).detach()

            data['img'][0].data[0] = adv_images
        return adv_images
