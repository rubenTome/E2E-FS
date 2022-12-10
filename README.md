# E2E-FS
E2E-FS: An End-to-End Feature Selection Method for Neural Networks

## Setup Instructions

### The python environment is included in the file requirements.txt.
Run the command:
 
    conda create --name e2efs --file ./requirements.txt

### Instructions for compiling the extern liblinear library:
Run the following commands:

    cd extern/liblinear
    make all 
    cd python
    make lib 

## Running the code
Activate the environment:

    conda activate e2efs 

All scripts are included into the script folder. To run it:

    PYTHONPATH=.:$PYTHONPATH python scripts/microarray/run_all.py
    PYTHONPATH=.:$PYTHONPATH python scripts/fs_challenge/run_all.py
    PYTHONPATH=.:$PYTHONPATH python scripts/deep/run_all.py

## HOW TO USE IT
Example included in example.py    

    ## LOAD DATA
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    ## LOAD MODEL AND COMPILE IT (NEVER FORGET TO COMPILE!)
    model = wrn164(input_shape=x_train.shape[1:], nclasses=10, regularization=5e-4)
    model.compile(optimizer='sgd', lr=1e-1, metrics=['acc'], loss='categorical_crossentropy')

    ## LOAD E2EFS AND RUN IT
    fs_class = models.E2EFSSoft(n_features_to_select=39).attach(model).fit(
        x_train, y_train, batch_size=128, validation_data=(x_test, y_test), verbose=2
    )

    ## FINE TUNING
    def scheduler(epoch):
        if epoch < 20:
            return .1
        elif epoch < 40:
            return .02
        elif epoch < 50:
            return .004
        else:
            return .0008

    fs_class.fine_tuning(x_train, y_train, epochs=60, batch_size=128, validation_data=(x_test, y_test),
                         callbacks=[LearningRateScheduler(scheduler)], verbose=2)
    print('FEATURE_RANKING :', fs_class.get_ranking())
    print('ACCURACY : ', fs_class.get_model().evaluate(x_test, y_test, batch_size=128)[-1])
    print('FEATURE_MASK NNZ :', np.count_nonzero(fs_class.get_mask()))