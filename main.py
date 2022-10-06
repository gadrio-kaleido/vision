import base64

import face_recognition
import numpy as np
from redis_om import Field, JsonModel, Migrator, get_redis_connection


class User(JsonModel):
    username: str = Field(index=True)
    encoded_bytes: str

    @property
    def encoded(self):
        value = base64.b64decode(self.encoded_bytes.encode())
        value = np.frombuffer(value)
        return value


Migrator().run()


def toRedis(r, value, nanme):
    """Store given Numpy array 'a' in Redis under key 'n'"""

    r.set(nanme, value.tobytes())
    return


def fromRedis(r, name):
    """Retrieve Numpy array from Redis key 'n'"""
    encoded = r.get(name)
    if encoded is None:
        return

    value = np.frombuffer(encoded)
    return value


r = get_redis_connection()

users = User.find(
    User.username == 'gonzaloadrio'
).all()

if users:
    user = users[0]

else:
    known_image = face_recognition.load_image_file('images/known_people/gonzalo.jpeg')
    known_encoding = face_recognition.face_encodings(known_image)[0]

    data_bytes = known_encoding.tobytes()
    encoded_str = base64.b64encode(data_bytes).decode()
    user = User(
        username='gonzaloadrio',
        encoded_bytes=encoded_str,
    )
    user.save()

users = User.find().all()

known_face_names, known_face_encodings = zip(*[(user, user.encoded) for user in users])

face_image = face_recognition.load_image_file("images/test/gonzalo.jpeg")
face_encoding = face_recognition.face_encodings(face_image)[0]

matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
best_match_index = np.argmin(face_distances)

user = 'unknown'
if matches[best_match_index]:
    user = known_face_names[best_match_index]

print(user)
