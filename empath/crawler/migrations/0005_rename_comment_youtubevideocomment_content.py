# Generated by Django 4.2.4 on 2023-09-09 08:09

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('crawler', '0004_youtubevideocomment_delete_youtubecomment'),
    ]

    operations = [
        migrations.RenameField(
            model_name='youtubevideocomment',
            old_name='comment',
            new_name='content',
        ),
    ]
