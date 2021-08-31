
import effdet.factory


def create_model(name='efficientdet_d0', data=None,
                 bench_task='train', pretrained=True,
                 checkpoint_path=''):

    assert data is not None
    model = effdet.factory.create_model(
        name,
        bench_task=bench_task,
        num_classes=data.num_classes,
        pretrained=pretrained,
        checkpoint_path=checkpoint_path,
        bench_labeler=True)

    return model
