# Generated by Django 4.2.4 on 2023-10-31 17:03

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('crawler', '0008_task'),
    ]

    operations = [
        migrations.AddField(
            model_name='task',
            name='user_id',
            field=models.IntegerField(null=True),
        ),
    ]
