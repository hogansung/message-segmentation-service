# message-segmentation-service

## Install Dependencies
### npm
```shell
npx create-react-app message-segmentor-service
```
To learn more about npm commands, please refer to [npm-README](./npm-README.md).

### flask
```shell
pip install flask
```

## Run Service Locally
One is suggested to launch the both of the following services in separated terminal tabs.

### npm
```shell
npm run start
```

### flask
```shell
FLASK_APP=service/message_segmentation.py FLASK_DEBUG=0 flask run
```
